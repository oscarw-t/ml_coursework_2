from sklearn.neighbors import NearestNeighbors


def compute_typicality(features, k=20):
        
    #typicality = 1 / mean_dist_to_k_nearest_neighbours
    #high score = point sits in a dense region

    k_actual = min(k, len(features) - 1)
    nn_model = NearestNeighbors(n_neighbors=k_actual + 1, metric='euclidean')
    nn_model.fit(features)
    distances, _ = nn_model.kneighbors(features)

    # distances[:, 0] is self-distance (0), skip it
    mean_dist = distances[:, 1:].mean(axis=1)
    return 1.0 / (mean_dist + 1e-10)