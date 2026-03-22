import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # save to file without a display
import matplotlib.pyplot as plt

from training.simclr_training import train_simclr
from training.feature_extraction import extract_features
from training.classifier import train_classifier
from training.linear_probe import extract_test_features, train_linear_probe
from typiclust.selection import typiclust_select_round, random_select_round
from typiclust.baselines import uncertainty_select_round

# Display labels, colours and markers for each strategy (Fig. 4/5 style)
_LABEL  = {'typiclust': 'TPC_RP', 'random': 'Random',
           'uncertainty': 'Uncertainty', 'margin': 'Margin', 'entropy': 'Entropy'}
_COLOR  = {'typiclust': 'tab:blue', 'random': 'black',
           'uncertainty': 'tab:red', 'margin': 'tab:green', 'entropy': 'tab:purple'}
_MARKER = {'typiclust': 'o', 'random': 's', 'uncertainty': '^',
           'margin': 'D', 'entropy': 'v'}


def _select_round(strategy, features, labels_gt, labeled_indices,
                  budget, device, classifier_epochs):
    n_total = len(labels_gt)
    if strategy == 'typiclust':
        return typiclust_select_round(features, labeled_indices, budget)
    elif strategy == 'random':
        return random_select_round(n_total, labeled_indices, budget)
    elif strategy in ('uncertainty', 'margin', 'entropy'):
        return uncertainty_select_round(
            labeled_indices, budget, n_total,
            strategy=strategy, device=device, epochs=classifier_epochs
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


def _print_table(results, num_rounds, budget_per_round, title):
    strats = list(results.keys())
    print(f"\n=== {title} ===")
    header = f"{'budget':<8}" + "".join(f"{s:>18}" for s in strats)
    print(header)
    print("-" * len(header))
    for r in range(num_rounds):
        budget = (r + 1) * budget_per_round
        row = f"{budget:<8}"
        for s in strats:
            vals = results[s][r]
            mean = np.mean(vals)
            se   = np.std(vals) / np.sqrt(len(vals))
            row += f"{mean:>10.2f}±{se:<6.2f}"
        print(row)


def _plot_results(results, num_rounds, budget_per_round, title, save_path):
    budgets = [(r + 1) * budget_per_round for r in range(num_rounds)]
    plt.figure(figsize=(7, 5))
    for strategy, round_data in results.items():
        means = [np.mean(round_data[r]) for r in range(num_rounds)]
        ses   = [np.std(round_data[r]) / np.sqrt(len(round_data[r]))
                 for r in range(num_rounds)]
        plt.errorbar(budgets, means, yerr=ses,
                     label=_LABEL.get(strategy, strategy),
                     color=_COLOR.get(strategy),
                     marker=_MARKER.get(strategy, 'o'),
                     capsize=3, linewidth=1.5)
    plt.xlabel("Cumulative budget (labeled examples)")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {save_path}")


def run_experiment(
    simclr_epochs=500,
    simclr_batch_size=512,
    budget_per_round=10,       
    num_rounds=5,
    classifier_epochs=100,      
    num_seeds=10,               
    device='cuda',
    checkpoint_path='simclr_checkpoint.pt',
    strategies=('typiclust', 'random', 'uncertainty', 'margin', 'entropy'),
):
    device = device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Step 1: SimCLR pre-training (loads from checkpoint if available)
    simclr_model = train_simclr(
        epochs=simclr_epochs, batch_size=simclr_batch_size,
        device=device, checkpoint_path=checkpoint_path
    )

    # Step 2: Feature extraction (train + test)
    train_features, train_labels = extract_features(simclr_model, device=device)
    test_features,  test_labels  = extract_test_features(simclr_model, device=device)
    print(f"  test features: {test_features.shape}\n")

    # Accumulators: results[strategy][round_idx] = list of accs over seeds
    sup_results   = {s: {r: [] for r in range(num_rounds)} for s in strategies}
    probe_results = {s: {r: [] for r in range(num_rounds)} for s in strategies}

    for seed in range(num_seeds):
        print(f"\n{'='*55}\nSeed {seed + 1}/{num_seeds}\n{'='*55}")
        np.random.seed(seed)
        torch.manual_seed(seed)

        for strategy in strategies:
            labeled_indices = []
            for round_idx in range(num_rounds):

                new_queries = _select_round(
                    strategy, train_features, train_labels,
                    labeled_indices, budget_per_round, device, classifier_epochs
                )
                labeled_indices = labeled_indices + new_queries
                budget = len(labeled_indices)

                # Framework (i): fully supervised (Appendix F.2.1)
                acc = train_classifier(
                    labeled_indices, device=device, epochs=classifier_epochs
                )

                # Framework (ii): linear probe on frozen SimCLR features
                # (Appendix F.2.2)
                probe_acc = train_linear_probe(
                    train_features[labeled_indices], train_labels[labeled_indices],
                    test_features, test_labels,
                    device=device, supervised_epochs=classifier_epochs
                )

                sup_results[strategy][round_idx].append(acc)
                probe_results[strategy][round_idx].append(probe_acc)

                print(f"  [{strategy:10s}] round {round_idx + 1} "
                      f"n={budget:3d}: sup={acc:.1f}%  probe={probe_acc:.1f}%")

    _print_table(sup_results,   num_rounds, budget_per_round, "Fully Supervised")
    _print_table(probe_results, num_rounds, budget_per_round,
                 "Self-Supervised Embedding (Linear Probe)")

    _plot_results(sup_results, num_rounds, budget_per_round,
                  title="Fully Supervised — CIFAR-10 (low budget)",
                  save_path="results_supervised.png")
    _plot_results(probe_results, num_rounds, budget_per_round,
                  title="Linear Probe on SimCLR features — CIFAR-10 (low budget)",
                  save_path="results_probe.png")

    return sup_results, probe_results


if __name__ == "__main__":
    run_experiment(
        simclr_epochs=200,
        simclr_batch_size=512,
        budget_per_round=10,
        num_rounds=5,
        classifier_epochs=100,
        num_seeds=5
        device='cuda',
        checkpoint_path='simclr_checkpoint.pt',
        strategies=('typiclust', 'random', 'uncertainty', 'hybrid'),
    )
