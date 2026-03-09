import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hcluster
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from flyvis_gnn.utils import fig_init, to_numpy


class EmbeddingCluster:
    def __init__(self, config):
        self.cluster_connectivity = config.training.cluster_connectivity    # 'single' (default) or 'average'

    def get(self, data, method, thresh=2.5):

        match method:
            case 'kmeans_auto':
                silhouette_avg_list = []
                silhouette_max = 0
                n_clusters = None
                for n in range(2, 10):
                    clusterer = KMeans(n_clusters=n, random_state=10, n_init='auto')
                    cluster_labels = clusterer.fit_predict(data)
                    if (np.unique(cluster_labels) == [0]):
                        n_clusters = 1
                    else:
                        silhouette_avg = silhouette_score(data, cluster_labels)
                        silhouette_avg_list.append(silhouette_avg)
                        if silhouette_avg > silhouette_max:
                            silhouette_max = silhouette_avg
                            n_clusters = n
                kmeans = KMeans(n_clusters=n_clusters, random_state=10, n_init='auto')
                k = kmeans.fit(data)
                clusters = k.labels_
            case 'distance':
                clusters = hcluster.fclusterdata(data, thresh, criterion="distance", method=self.cluster_connectivity) - 1
                n_clusters = len(np.unique(clusters))
            case 'inconsistent':
                clusters = hcluster.fclusterdata(data, thresh, criterion="inconsistent", method=self.cluster_connectivity) - 1
                n_clusters = len(np.unique(clusters))

            case _:
                raise ValueError(f'Unknown method {method}')

        return clusters, n_clusters


def sparsify_cluster(cluster_method, proj_interaction, embedding, cluster_distance_threshold, type_list, n_neuron_types, embedding_cluster):

    # normalization of projection because UMAP output is not normalized
    proj_interaction = (proj_interaction - np.min(proj_interaction)) / (np.max(proj_interaction) - np.min(proj_interaction)+1e-10)
    embedding = (embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding)+1e-10)

    match cluster_method:
        case 'kmeans_auto_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
        case 'kmeans_auto_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
            proj_interaction = embedding
        case 'distance_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance', thresh=cluster_distance_threshold)
        case 'distance_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=cluster_distance_threshold)
            proj_interaction = embedding
        case 'inconsistent_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'inconsistent', thresh=cluster_distance_threshold)
        case 'inconsistent_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'inconsistent', thresh=cluster_distance_threshold)
            proj_interaction = embedding
        case 'distance_both':
            new_projection = np.concatenate((proj_interaction, embedding), axis=-1)
            labels, n_clusters = embedding_cluster.get(new_projection, 'distance', thresh=cluster_distance_threshold)

    label_list = []
    for n in range(n_neuron_types):
        pos = torch.argwhere(type_list == n)
        pos = to_numpy(pos)
        if len(pos) > 0:
            tmp = labels[pos[:,0]]
            label_list.append(np.round(np.median(tmp)))
            np.argwhere(labels == np.median(tmp))

    label_list = np.array(label_list)

    fig,ax = fig_init()
    for n in label_list:
        pos = np.argwhere(labels == n)
        # print(len(pos))
        if len(pos) > 0:
            ax.scatter(embedding[pos, 0], embedding[pos, 1], s=5)
    plt.close()

    new_labels = np.ones_like(labels) * n_neuron_types
    for n in range(n_neuron_types):
        if n < len(label_list):
            new_labels[labels == label_list[n]] = n

    fig,ax = fig_init()
    ax.scatter(proj_interaction[:, 0], proj_interaction[:, 1], c=new_labels, s=5, cmap='tab20')
    plt.close()

    return labels, n_clusters, new_labels

def sparsify_cluster_state(cluster_method, proj_interaction, embedding, cluster_distance_threshold, true_type_list, n_neuron_types, embedding_cluster):

    # normalization of projection because UMAP output is not normalized
    proj_interaction = (proj_interaction - np.min(proj_interaction)) / (np.max(proj_interaction) - np.min(proj_interaction)+1e-10)
    embedding = (embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding)+1e-10)

    start_time = time.time()
    match cluster_method:
        case 'kmeans_auto_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
        case 'kmeans_auto_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
            proj_interaction = embedding
        case 'distance_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance', thresh=cluster_distance_threshold)
        case 'distance_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=cluster_distance_threshold)
            proj_interaction = embedding
        case 'inconsistent_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'inconsistent', thresh=cluster_distance_threshold)
        case 'inconsistent_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'inconsistent', thresh=cluster_distance_threshold)
            proj_interaction = embedding
        case 'distance_both':
            new_projection = np.concatenate((proj_interaction, embedding), axis=-1)
            labels, n_clusters = embedding_cluster.get(new_projection, 'distance', thresh=cluster_distance_threshold)

    computation_time = time.time() - start_time
    print(f"clustering computation time is {computation_time:0.2f} seconds.")

    label_list = []
    for n in range(n_neuron_types):
        pos = np.argwhere(true_type_list == n).squeeze().astype(int)
        if len(pos)>0:
            tmp = labels[pos]
            label_list.append(np.round(np.median(tmp)))
        else:
            label_list.append(0)
    label_list = np.array(label_list)
    new_labels = np.ones_like(labels) * n_neuron_types
    for n in range(n_neuron_types):
        new_labels[labels == label_list[n]] = n

    return labels, n_clusters, new_labels

def evaluate_embedding_clustering(model, type_list, n_types=64):
    """
    Cluster model.a embeddings and evaluate against true neuron types
    """

    # Extract embeddings
    embeddings = to_numpy(model.a)
    true_labels = to_numpy(type_list).flatten()  # Fix: add .flatten()

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_types, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Calculate metrics that don't require label alignment
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)

    # Calculate accuracy with optimal label mapping (Hungarian algorithm)
    def find_optimal_mapping(true_labels, cluster_labels, n_clusters):
        # Create confusion matrix
        confusion_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(len(true_labels)):
            confusion_matrix[int(true_labels[i]), int(cluster_labels[i])] += 1  # Add int() for safety

        # Find optimal assignment using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

        # Create mapping dictionary
        mapping = {col_ind[i]: row_ind[i] for i in range(len(col_ind))}

        # Map cluster labels to true labels
        mapped_labels = np.array([mapping.get(label, -1) for label in cluster_labels])

        return mapped_labels

    # Get optimally mapped labels and calculate accuracy
    mapped_labels = find_optimal_mapping(true_labels, cluster_labels, n_types)
    accuracy = accuracy_score(true_labels, mapped_labels)

    return {
        'ari': ari_score,
        'nmi': nmi_score,
        'accuracy': accuracy,
        'cluster_labels': cluster_labels,
        'mapped_labels': mapped_labels
    }

def clustering_evaluation(data, type_list, eps=0.5):
    """
    Blind clustering using DBSCAN (doesn't require number of clusters)
    Parameter: eps - maximum distance between points in same cluster
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score

    true_labels = to_numpy(type_list).flatten()

    # Perform DBSCAN clustering (automatically finds number of clusters)
    dbscan = DBSCAN(eps=eps, min_samples=5)
    cluster_labels = dbscan.fit_predict(data)

    # Count found clusters (excluding noise points labeled as -1)
    n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise_points = list(cluster_labels).count(-1)

    # Handle noise points for metrics (assign them to separate cluster)
    cluster_labels_clean = cluster_labels.copy()
    cluster_labels_clean[cluster_labels_clean == -1] = n_clusters_found  # Assign noise to separate cluster

    # Calculate clustering metrics
    ari_score = adjusted_rand_score(true_labels, cluster_labels_clean)
    nmi_score = normalized_mutual_info_score(true_labels, cluster_labels_clean)

    # Calculate accuracy with optimal label mapping (Hungarian algorithm)
    def find_optimal_mapping_blind(true_labels, cluster_labels):
        n_true_clusters = len(np.unique(true_labels))
        n_found_clusters = len(np.unique(cluster_labels))

        # Create confusion matrix
        confusion_matrix = np.zeros((n_true_clusters, n_found_clusters))
        for i in range(len(true_labels)):
            true_idx = int(true_labels[i])
            cluster_idx = int(cluster_labels[i])
            if 0 <= true_idx < n_true_clusters and 0 <= cluster_idx < n_found_clusters:
                confusion_matrix[true_idx, cluster_idx] += 1

        # Find optimal assignment using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

        # Create mapping dictionary
        mapping = {}
        for i in range(len(col_ind)):
            if i < len(row_ind):
                mapping[col_ind[i]] = row_ind[i]

        # Map cluster labels to true labels
        mapped_labels = np.array([mapping.get(label, -1) for label in cluster_labels])

        return mapped_labels

    # Calculate accuracy
    mapped_labels = find_optimal_mapping_blind(true_labels, cluster_labels_clean)
    accuracy = accuracy_score(true_labels, mapped_labels)

    # Calculate silhouette score (clustering quality)
    from sklearn.metrics import silhouette_score
    if n_clusters_found > 1:
        silhouette = silhouette_score(data, cluster_labels_clean)
    else:
        silhouette = 0.0

    return {
        'n_clusters_found': n_clusters_found,
        'n_noise_points': n_noise_points,
        'eps_used': eps,
        'ari': ari_score,
        'nmi': nmi_score,
        'accuracy': accuracy,
        'silhouette': silhouette,
        'cluster_labels': cluster_labels,
        'mapped_labels': mapped_labels,
        'total_points': len(data)
    }

def functional_clustering_evaluation(func_list, type_list, eps=0.5, min_samples=5, normalize=True):
    """
    Cluster neurons based on their phi function responses instead of embeddings

    Parameters:
    - func_list: List of torch tensors, each containing phi function output for one neuron
    - type_list: True neuron type labels
    - eps: DBSCAN epsilon parameter
    - min_samples: DBSCAN min_samples parameter
    - normalize: Whether to normalize function responses
    """

    if isinstance(func_list, torch.Tensor):
        # func_list is already a tensor
        func_features = to_numpy(func_list)
    elif isinstance(func_list, list) and len(func_list) > 0:
        if isinstance(func_list[0], torch.Tensor):
            # Stack all functions into single array
            func_array = torch.stack(func_list).squeeze()  # Shape: (n_neurons, n_points)
            func_features = to_numpy(func_array)
            print("Stacked list of tensors")
        else:
            func_features = np.array(func_list)
            print("Converted list to numpy array")
    else:
        raise ValueError(f"Unexpected func_list type: {type(func_list)}")

    # Handle different function output shapes
    if len(func_features.shape) == 3:
        # If functions are (n_neurons, n_points, 1), flatten last dimension
        func_features = func_features.squeeze(-1)


    # Normalize functions if requested
    if normalize:
        scaler = StandardScaler()
        func_features_processed = scaler.fit_transform(func_features)
    else:
        func_features_processed = func_features

    # Extract true labels
    true_labels = to_numpy(type_list).flatten()

    # Perform DBSCAN clustering on functional responses
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(func_features_processed)

    # Count clusters and noise points
    unique_clusters = np.unique(cluster_labels)
    n_clusters_found = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    n_noise_points = np.sum(cluster_labels == -1)

    # Handle noise points for metrics (assign to separate cluster)
    cluster_labels_clean = cluster_labels.copy()
    cluster_labels_clean[cluster_labels_clean == -1] = n_clusters_found  # Assign noise to separate cluster

    # Calculate clustering metrics
    ari_score = adjusted_rand_score(true_labels, cluster_labels_clean)
    nmi_score = normalized_mutual_info_score(true_labels, cluster_labels_clean)

    # Calculate silhouette score (functional clustering quality)
    if n_clusters_found > 1 and n_noise_points < len(cluster_labels):
        # Use original cluster labels for silhouette (excluding noise points)
        valid_indices = cluster_labels != -1
        if np.sum(valid_indices) > 1:
            silhouette = silhouette_score(
                func_features_processed[valid_indices],
                cluster_labels[valid_indices]
            )
        else:
            silhouette = 0.0
    else:
        silhouette = 0.0

    # Calculate accuracy with optimal mapping (Hungarian algorithm)
    def find_optimal_functional_mapping(true_labels, cluster_labels):
        n_true_clusters = len(np.unique(true_labels))
        n_found_clusters = len(np.unique(cluster_labels))

        # Create confusion matrix
        confusion_matrix = np.zeros((n_true_clusters, n_found_clusters))
        for i in range(len(true_labels)):
            true_idx = int(true_labels[i])
            cluster_idx = int(cluster_labels[i])
            if 0 <= true_idx < n_true_clusters and 0 <= cluster_idx < n_found_clusters:
                confusion_matrix[true_idx, cluster_idx] += 1

        # Find optimal assignment using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

        # Create mapping dictionary
        mapping = {}
        for i in range(len(col_ind)):
            if i < len(row_ind):
                mapping[col_ind[i]] = row_ind[i]

        # Map cluster labels to true labels
        mapped_labels = np.array([mapping.get(label, -1) for label in cluster_labels])

        return mapped_labels, confusion_matrix

    # Get optimal mapping and calculate accuracy
    mapped_labels, confusion_matrix = find_optimal_functional_mapping(true_labels, cluster_labels_clean)
    accuracy = accuracy_score(true_labels, mapped_labels)

    # Calculate additional functional clustering metrics
    def calculate_functional_purity():
        """Calculate how 'pure' each functional cluster is in terms of true types"""
        cluster_purities = []
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise
                continue
            cluster_mask = cluster_labels == cluster_id
            cluster_true_types = true_labels[cluster_mask]
            if len(cluster_true_types) > 0:
                # Purity = fraction of most common true type in this cluster
                unique_types, counts = np.unique(cluster_true_types, return_counts=True)
                purity = np.max(counts) / len(cluster_true_types)
                cluster_purities.append(purity)

        return np.mean(cluster_purities) if cluster_purities else 0.0

    def calculate_functional_completeness():
        """Calculate how completely each true type is captured by functional clusters"""
        type_completeness = []
        for true_type in np.unique(true_labels):
            type_mask = true_labels == true_type
            type_clusters = cluster_labels[type_mask]
            type_clusters_clean = type_clusters[type_clusters != -1]  # Exclude noise

            if len(type_clusters_clean) > 0:
                # Completeness = fraction in largest cluster for this type
                unique_clusters_for_type, counts = np.unique(type_clusters_clean, return_counts=True)
                completeness = np.max(counts) / len(type_clusters_clean)
                type_completeness.append(completeness)

        return np.mean(type_completeness) if type_completeness else 0.0

    purity = calculate_functional_purity()
    completeness = calculate_functional_completeness()

    # Calculate function diversity within vs between clusters
    def calculate_functional_separation():
        """Calculate how well separated functional clusters are"""
        if n_clusters_found <= 1:
            return 0.0, 0.0, 0.0

        within_cluster_distances = []
        between_cluster_distances = []

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise
                continue

            cluster_mask = cluster_labels == cluster_id
            cluster_functions = func_features_processed[cluster_mask]

            if len(cluster_functions) > 1:
                # Calculate pairwise distances within cluster
                from sklearn.metrics.pairwise import euclidean_distances
                within_distances = euclidean_distances(cluster_functions)
                # Take upper triangle (avoid diagonal and duplicates)
                upper_triangle = np.triu_indices_from(within_distances, k=1)
                within_cluster_distances.extend(within_distances[upper_triangle])

            # Calculate distances to other clusters
            other_cluster_mask = (cluster_labels != cluster_id) & (cluster_labels != -1)
            if np.any(other_cluster_mask):
                other_functions = func_features_processed[other_cluster_mask]
                between_distances = euclidean_distances(cluster_functions, other_functions)
                between_cluster_distances.extend(between_distances.flatten())

        avg_within = np.mean(within_cluster_distances) if within_cluster_distances else 0.0
        avg_between = np.mean(between_cluster_distances) if between_cluster_distances else 0.0
        separation_ratio = avg_between / avg_within if avg_within > 0 else 0.0

        return avg_within, avg_between, separation_ratio

    avg_within_dist, avg_between_dist, separation_ratio = calculate_functional_separation()

    return {
        'n_clusters_found': n_clusters_found,
        'n_noise_points': n_noise_points,
        'eps_used': eps,
        'min_samples_used': min_samples,
        'ari': ari_score,
        'nmi': nmi_score,
        'accuracy': accuracy,
        'silhouette': silhouette,
        'purity': purity,
        'completeness': completeness,
        'avg_within_cluster_distance': avg_within_dist,
        'avg_between_cluster_distance': avg_between_dist,
        'separation_ratio': separation_ratio,
        'cluster_labels': cluster_labels,
        'mapped_labels': mapped_labels,
        'confusion_matrix': confusion_matrix,
        'total_points': len(func_features),
        'normalization_applied': normalize,
        'function_features_shape': func_features.shape
    }



def clustering_evaluation_augmented(data, type_list, eps=0.5):
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    true_labels = to_numpy(type_list).flatten()
    dbscan = DBSCAN(eps=eps, min_samples=5)
    cluster_labels = dbscan.fit_predict(data)
    n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise_points = list(cluster_labels).count(-1)
    cluster_labels_clean = cluster_labels.copy()
    cluster_labels_clean[cluster_labels_clean == -1] = n_clusters_found

    n_true = len(np.unique(true_labels))
    n_found = len(np.unique(cluster_labels_clean))
    conf_mat = np.zeros((n_true, n_found))
    for i in range(len(true_labels)):
        conf_mat[int(true_labels[i]), int(cluster_labels_clean[i])] += 1
    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    mapping = {col_ind[i]: row_ind[i] for i in range(len(col_ind)) if i < len(row_ind)}
    mapped_labels = np.array([mapping.get(label, -1) for label in cluster_labels_clean])

    accuracy = accuracy_score(true_labels, mapped_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels_clean)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels_clean)
    sil = silhouette_score(data, cluster_labels_clean) if n_clusters_found > 1 else 0.0

    return {'n_clusters_found': n_clusters_found, 'n_noise_points': n_noise_points, 'eps_used': eps,
            'ari': ari, 'nmi': nmi, 'accuracy': accuracy, 'silhouette': sil}

def clustering_spectral(data, type_list, n_clusters=None):
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    true_labels = to_numpy(type_list).flatten()
    if n_clusters is None:
        n_clusters = len(np.unique(true_labels))

    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=10, random_state=42)
    cluster_labels = spectral.fit_predict(data)

    n_true = len(np.unique(true_labels))
    n_found = len(np.unique(cluster_labels))
    conf_mat = np.zeros((n_true, n_found))
    for i in range(len(true_labels)):
        conf_mat[int(true_labels[i]), int(cluster_labels[i])] += 1
    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    mapping = {col_ind[i]: row_ind[i] for i in range(len(col_ind))}
    mapped_labels = np.array([mapping.get(label, -1) for label in cluster_labels])

    accuracy = accuracy_score(true_labels, mapped_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    sil = silhouette_score(data, cluster_labels) if n_found > 1 else 0.0

    return {'n_clusters': n_clusters, 'accuracy': accuracy, 'ari': ari, 'nmi': nmi, 'silhouette': sil}

def clustering_hdbscan(data, type_list, min_cluster_size=5):
    from hdbscan import HDBSCAN
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    true_labels = to_numpy(type_list).flatten()
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5)
    cluster_labels = hdb.fit_predict(data)

    n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    cluster_labels_clean = cluster_labels.copy()
    cluster_labels_clean[cluster_labels_clean == -1] = n_clusters_found

    n_true = len(np.unique(true_labels))
    n_found = len(np.unique(cluster_labels_clean))
    conf_mat = np.zeros((n_true, n_found))
    for i in range(len(true_labels)):
        conf_mat[int(true_labels[i]), int(cluster_labels_clean[i])] += 1
    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    mapping = {col_ind[i]: row_ind[i] for i in range(len(col_ind))}
    mapped_labels = np.array([mapping.get(label, -1) for label in cluster_labels_clean])

    accuracy = accuracy_score(true_labels, mapped_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels_clean)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels_clean)
    sil = silhouette_score(data, cluster_labels_clean) if n_clusters_found > 1 else 0.0

    return {'n_clusters_found': n_clusters_found, 'min_cluster_size': min_cluster_size,
            'accuracy': accuracy, 'ari': ari, 'nmi': nmi, 'silhouette': sil}

def clustering_gmm(data, type_list, n_components=None):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from sklearn.mixture import GaussianMixture

    true_labels = to_numpy(type_list).flatten()
    if n_components is None:
        n_components = len(np.unique(true_labels))

    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    cluster_labels = gmm.fit_predict(data)

    # Fix: Ensure cluster labels are contiguous starting from 0
    unique_clusters = np.unique(cluster_labels)
    cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_clusters)}
    cluster_labels_remapped = np.array([cluster_mapping[label] for label in cluster_labels])

    n_true = len(np.unique(true_labels))
    n_found = len(unique_clusters)  # Use actual number of unique clusters

    conf_mat = np.zeros((n_true, n_found))
    for i in range(len(true_labels)):
        try:
            true_idx = int(true_labels[i])
            cluster_idx = int(cluster_labels_remapped[i])
            if 0 <= true_idx < n_true and 0 <= cluster_idx < n_found:
                conf_mat[true_idx, cluster_idx] += 1
        except (IndexError, ValueError):
            print(f"Skipping invalid indices: true_idx={true_labels[i]}, cluster_idx={cluster_labels[i]}")
            continue

    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    mapping = {col_ind[i]: row_ind[i] for i in range(len(col_ind))}
    mapped_labels = np.array([mapping.get(label, -1) for label in cluster_labels_remapped])

    accuracy = accuracy_score(true_labels, mapped_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels_remapped)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels_remapped)
    sil = silhouette_score(data, cluster_labels_remapped) if n_found > 1 else 0.0

    return {'n_components': n_components, 'accuracy': accuracy, 'ari': ari, 'nmi': nmi, 'silhouette': sil}


def connectivity_stats(w, src, dst, n):
    """Per-neuron mean/std of in-weights and out-weights."""
    in_count = np.bincount(dst, minlength=n).astype(np.float64)
    out_count = np.bincount(src, minlength=n).astype(np.float64)
    in_sum = np.bincount(dst, weights=w, minlength=n)
    out_sum = np.bincount(src, weights=w, minlength=n)
    in_sq = np.bincount(dst, weights=w ** 2, minlength=n)
    out_sq = np.bincount(src, weights=w ** 2, minlength=n)
    safe_in = np.where(in_count > 0, in_count, 1)
    safe_out = np.where(out_count > 0, out_count, 1)
    in_mean = in_sum / safe_in
    out_mean = out_sum / safe_out
    in_std = np.sqrt(np.maximum(in_sq / safe_in - in_mean ** 2, 0))
    out_std = np.sqrt(np.maximum(out_sq / safe_out - out_mean ** 2, 0))
    in_mean[in_count == 0] = 0
    out_mean[out_count == 0] = 0
    in_std[in_count == 0] = 0
    out_std[out_count == 0] = 0
    return in_mean, in_std, out_mean, out_std


def reinit_mlp_weights(mlp):
    """Reinitialize all Linear layers of an MLP to match MLP.__init__ scheme.

    Hidden and output layers: normal_(std=0.1) for weights, zeros_ for biases.
    """
    import torch.nn as nn
    for layer in mlp.layers:
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, std=0.1)
            nn.init.zeros_(layer.bias)


def umap_cluster_reassign(model, config, x_ts, edges, n_neurons, type_list, device,
                          logger=None, reinit_mlps=False, relearn_epochs=0):
    """UMAP-based clustering on augmented features (W, tau, embedding, Vrest).

    Builds augmented feature vector from model parameters, reduces with UMAP,
    clusters in UMAP space, reassigns embeddings to cluster medians, and
    normalizes to [0, 1].  Optionally relearns g_phi/f_theta to match
    cluster-median function shapes (like ParticleGraph replace_embedding_function).

    Returns dict with cluster info (n_clusters, accuracy, labels).
    """
    import warnings

    import umap
    import umap.umap_ as _umap_mod
    # Fix umap / scikit-learn >=1.6 incompatibility
    try:
        import sklearn.utils.validation as _skval
        if not hasattr(_skval, '_orig_check_array'):
            _skval._orig_check_array = _skval.check_array
            def _check_array_compat(*args, **kwargs):
                kwargs.pop('force_all_finite', None)
                return _skval._orig_check_array(*args, **kwargs)
            _skval.check_array = _check_array_compat
            if hasattr(_umap_mod, 'check_array'):
                _umap_mod.check_array = _check_array_compat
    except Exception:
        pass

    from flyvis_gnn.metrics import compute_activity_stats, extract_f_theta_slopes

    tc = config.training

    # 1. Extract tau and V_rest from f_theta slopes
    mu, sigma = compute_activity_stats(x_ts, device)
    slopes, offsets = extract_f_theta_slopes(model, config, n_neurons, mu, sigma, device)
    learned_V_rest = np.where(slopes != 0, -offsets / slopes, 1.0)[:n_neurons]
    learned_tau = np.where(slopes != 0, 1.0 / -slopes, 1.0)[:n_neurons]
    learned_tau = np.clip(learned_tau, 0, 1)

    # 2. Compute per-neuron W statistics
    learned_W = to_numpy(model.W.squeeze())
    src = to_numpy(edges[0])
    dst = to_numpy(edges[1])
    w_in_mean, w_in_std, w_out_mean, w_out_std = connectivity_stats(
        learned_W, src, dst, n_neurons)

    # 3. Build augmented feature vector
    embedding = to_numpy(model.a.squeeze())
    a_aug = np.column_stack([embedding, learned_tau, learned_V_rest,
                             w_in_mean, w_in_std, w_out_mean, w_out_std])

    # 4. UMAP dimensionality reduction
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reducer = umap.UMAP(n_components=2,
                            n_neighbors=tc.umap_cluster_n_neighbors,
                            min_dist=tc.umap_cluster_min_dist,
                            random_state=tc.seed)
        a_umap = reducer.fit_transform(a_aug)

    # 5. Cluster in UMAP space
    if tc.umap_cluster_method == 'dbscan':
        labels = DBSCAN(eps=tc.umap_cluster_eps, min_samples=5).fit_predict(a_umap)
    elif tc.umap_cluster_method == 'gmm':
        from sklearn.mixture import GaussianMixture
        labels = GaussianMixture(n_components=tc.umap_cluster_gmm_n,
                                 random_state=tc.seed).fit_predict(a_umap)
    else:
        return None

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # 5b. Evaluate g_phi and f_theta before reassigning embeddings (for relearning targets)
    func_list_edge = None
    func_list_phi = None
    if relearn_epochs > 0 and n_clusters > 0:
        n_pts = 1000
        rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], n_pts, device=device)
        model_name = config.graph_model.signal_model_name
        g_phi_positive = config.graph_model.g_phi_positive

        with torch.no_grad():
            # g_phi: input = [v_j, a_j] for flyvis_A
            func_list_edge = torch.zeros(n_neurons, n_pts, device=device)
            func_list_phi = torch.zeros(n_neurons, n_pts, device=device)
            batch_size = 500
            for start in range(0, n_neurons, batch_size):
                end = min(start + batch_size, n_neurons)
                n_batch = end - start
                rr_batch = rr.unsqueeze(0).expand(n_batch, -1)  # (B, n_pts)
                rr_flat = rr_batch.reshape(-1, 1)  # (B*n_pts, 1)
                emb = model.a[start:end]  # (B, emb_dim)
                emb_flat = emb.unsqueeze(1).expand(-1, n_pts, -1).reshape(-1, emb.shape[1])

                # g_phi
                if 'flyvis_B' in model_name:
                    edge_in = torch.cat([rr_flat * 0, rr_flat, emb_flat, emb_flat], dim=1)
                else:
                    edge_in = torch.cat([rr_flat, emb_flat], dim=1)
                edge_out = model.g_phi(edge_in.float())
                if g_phi_positive:
                    edge_out = edge_out ** 2
                func_list_edge[start:end] = edge_out[:, 0].reshape(n_batch, n_pts)

                # f_theta: input = [v, a, msg=0, exc=0]
                zeros_flat = torch.zeros_like(rr_flat)
                phi_in = torch.cat([rr_flat, emb_flat, zeros_flat, zeros_flat], dim=1)
                phi_out = model.f_theta(phi_in.float())
                func_list_phi[start:end] = phi_out[:, 0].reshape(n_batch, n_pts)

        # Compute target functions: median per cluster
        y_func_edge = func_list_edge.clone()
        y_func_phi = func_list_phi.clone()
        for label in np.unique(labels):
            if label == -1:
                continue
            indices = np.where(labels == label)[0]
            if len(indices) > 0:
                target_edge = torch.median(func_list_edge[indices], dim=0).values
                target_phi = torch.median(func_list_phi[indices], dim=0).values
                y_func_edge[indices] = target_edge
                y_func_phi[indices] = target_phi

    # 6. Replace embeddings with UMAP projections, then take cluster medians
    a_umap_tensor = torch.tensor(a_umap, dtype=model.a.dtype, device=model.a.device)
    with torch.no_grad():
        model.a.data[:n_neurons] = a_umap_tensor
        for label in np.unique(labels):
            if label == -1:
                continue
            indices = np.where(labels == label)[0]
            if len(indices) > 1:
                cluster_median = torch.median(model.a[indices], dim=0).values
                model.a[indices] = cluster_median

    # 7. Normalize embeddings to [0, 1]
    with torch.no_grad():
        a_min = model.a.min(dim=0).values
        a_max = model.a.max(dim=0).values
        model.a.data = (model.a.data - a_min) / (a_max - a_min + 1e-10)

    # 8. Optionally reinitialize f_theta and g_phi
    if reinit_mlps:
        reinit_mlp_weights(model.f_theta)
        reinit_mlp_weights(model.g_phi)
        msg = "Reinitialized f_theta and g_phi weights"
        print(msg)
        if logger:
            logger.info(msg)

    # 9. Relearn g_phi and f_theta to match cluster-median function shapes
    if relearn_epochs > 0 and func_list_edge is not None:
        from tqdm import trange
        n_pts = 1000
        rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], n_pts, device=device)
        model_name = config.graph_model.signal_model_name
        g_phi_positive = config.graph_model.g_phi_positive

        # Temporary optimizer: freeze embedding, train only g_phi and f_theta
        relearn_params = list(model.g_phi.parameters()) + list(model.f_theta.parameters())
        relearn_optimizer = torch.optim.Adam(relearn_params, lr=tc.learning_rate_start)

        for sub_epoch in trange(relearn_epochs, ncols=100, desc='relearn MLP'):
            relearn_optimizer.zero_grad()
            loss_edge = torch.tensor(0.0, device=device)
            loss_phi = torch.tensor(0.0, device=device)

            batch_size = 500
            for start in range(0, n_neurons, batch_size):
                end = min(start + batch_size, n_neurons)
                n_batch = end - start
                rr_batch = rr.unsqueeze(0).expand(n_batch, -1)
                rr_flat = rr_batch.reshape(-1, 1)
                emb = model.a[start:end].detach()
                emb_flat = emb.unsqueeze(1).expand(-1, n_pts, -1).reshape(-1, emb.shape[1])

                # g_phi forward
                if 'flyvis_B' in model_name:
                    edge_in = torch.cat([rr_flat * 0, rr_flat, emb_flat, emb_flat], dim=1)
                else:
                    edge_in = torch.cat([rr_flat, emb_flat], dim=1)
                edge_out = model.g_phi(edge_in.float())
                if g_phi_positive:
                    edge_out = edge_out ** 2
                pred_edge = edge_out[:, 0].reshape(n_batch, n_pts)
                loss_edge = loss_edge + (pred_edge - y_func_edge[start:end].detach()).norm(2)

                # f_theta forward
                zeros_flat = torch.zeros_like(rr_flat)
                phi_in = torch.cat([rr_flat, emb_flat, zeros_flat, zeros_flat], dim=1)
                phi_out = model.f_theta(phi_in.float())
                pred_phi = phi_out[:, 0].reshape(n_batch, n_pts)
                loss_phi = loss_phi + (pred_phi - y_func_phi[start:end].detach()).norm(2)

            total_loss = loss_edge + loss_phi
            total_loss.backward()
            relearn_optimizer.step()

        msg = f"relearn MLP: {relearn_epochs} epochs, final loss edge={loss_edge.item()/n_neurons:.4f} phi={loss_phi.item()/n_neurons:.4f}"
        print(msg)
        if logger:
            logger.info(msg)

    # 10. Evaluate clustering quality against ground truth types
    true_labels = to_numpy(type_list).flatten()
    labels_clean = labels.copy()
    labels_clean[labels_clean == -1] = n_clusters
    ari = adjusted_rand_score(true_labels, labels_clean)
    accuracy = 0.0
    if n_clusters > 0:
        n_true = len(np.unique(true_labels))
        n_found = len(np.unique(labels_clean))
        conf_mat = np.zeros((n_true, n_found))
        for i in range(len(true_labels)):
            ti = int(true_labels[i])
            ci = int(labels_clean[i])
            if 0 <= ti < n_true and 0 <= ci < n_found:
                conf_mat[ti, ci] += 1
        row_ind, col_ind = linear_sum_assignment(-conf_mat)
        mapping = {col_ind[j]: row_ind[j] for j in range(len(col_ind))}
        mapped = np.array([mapping.get(l, -1) for l in labels_clean])
        accuracy = accuracy_score(true_labels, mapped)

    msg = (f"UMAP cluster: {n_clusters} clusters, accuracy={accuracy:.3f}, ARI={ari:.3f}")
    print(msg)
    if logger:
        logger.info(msg)

    return {
        'n_clusters': n_clusters,
        'accuracy': accuracy,
        'ari': ari,
        'labels': labels,
        'a_umap': a_umap,
    }


# Usage example:
# After running your phi function plotting code:
# results = functional_clustering_evaluation(func_list, type_list, eps=0.2)
# comparison = compare_embedding_vs_functional_clustering(model, type_list, func_list)
if __name__ == '__main__':
    # generate 3 clusters of each around 100 points and one orphan point
    from types import SimpleNamespace

    # Create a mock config for testing
    mock_config = SimpleNamespace(training=SimpleNamespace(cluster_connectivity='single'))
    embedding_cluster = EmbeddingCluster(mock_config)

    N = 100
    data = np.random.randn(3 * N, 2)
    data[:N] += 5
    data[-N:] += 10
    data[-1:] -= 20

    # clustering
    thresh = 1.5
    clusters, n_clusters = embedding_cluster.get(data, method="distance")

    # plotting
    plt.scatter(*np.transpose(data), c=clusters, s=5)
    plt.axis("equal")
    title = "threshold: %f, number of clusters: %d" % (thresh, n_clusters)
    plt.title(title)
    plt.show()
