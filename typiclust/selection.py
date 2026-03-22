import numpy as np

from typiclust.typicality import compute_typicality
from typiclust.k_means import cluster


def typiclust_select_round(features, labeled_indices, budget,
                            max_clusters=500, k_typicality=20):
    """One round of TypiClust selection (Algorithm 1 + Appendix F.1, Step 3).

    Fixes vs original:
      - features[cluster] (TypeError: cluster is a function) -> features[members]
      - k_actual used len(unlabeled) -> now uses len(members) (full cluster size)
      - typicality computed on unlabeled subset -> now on all cluster members,
        scores then filtered to unlabeled candidates per Appendix F.1 Step 3.
    """
    N = len(features)
    n_clusters = min(len(labeled_indices) + budget, max_clusters)

    cluster_assignments = cluster(features, n_clusters)

    labeled_set = set(labeled_indices)
    cluster_members = {}
    cluster_label_count = {}

    for i in range(N):
        cid = cluster_assignments[i]
        cluster_members.setdefault(cid, []).append(i)
        cluster_label_count.setdefault(cid, 0)
        if i in labeled_set:
            cluster_label_count[cid] += 1

    queries = []
    temp_labeled = set(labeled_indices)

    queries = []
    temp_labeled = set(labeled_indices)

    while len(queries) < budget:
        eligible = {cid: m for cid, m in cluster_members.items() if len(m) >= 5}
        if not eligible:
            break

        min_count = min(cluster_label_count.get(cid, 0) for cid in eligible)
        candidates = {cid: m for cid, m in eligible.items()
                      if cluster_label_count.get(cid, 0) == min_count}
        best_cluster = max(candidates, key=lambda cid: len(candidates[cid]))

        members = cluster_members[best_cluster]
        unlabeled = [i for i in members if i not in temp_labeled]

        if not unlabeled:
            cluster_label_count[best_cluster] = float('inf')
            continue  # now safely retries without losing a query slot

        k_actual = min(k_typicality, len(members))
        all_scores = compute_typicality(features[members], k=k_actual)
        member_score = {idx: s for idx, s in zip(members, all_scores)}
        unlabeled_scores = np.array([member_score[i] for i in unlabeled])
        selected = unlabeled[int(np.argmax(unlabeled_scores))]

        queries.append(selected)
        temp_labeled.add(selected)
        cluster_label_count[best_cluster] += 1

    return queries


def random_select_round(n_total, labeled_indices, budget):
    """Uniform random selection from the unlabeled pool."""
    remaining = list(set(range(n_total)) - set(labeled_indices))
    np.random.shuffle(remaining)
    return remaining[:budget]

def hybrid_select_round(features, labeled_indices, budget, round_idx,
                        n_total=50000, device='cuda', classifier_epochs=100,
                        switch_round=3):
    """Phase-transition hybrid: TypiClust early, uncertainty late."""
    if round_idx < switch_round:
        return typiclust_select_round(features, labeled_indices, budget)
    else:
        from typiclust.baselines import uncertainty_select_round
        return uncertainty_select_round(
            labeled_indices, budget, n_total,
            strategy='uncertainty', device=device, epochs=classifier_epochs
        )