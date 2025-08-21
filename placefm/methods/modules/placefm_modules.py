import torch
from torch_sparse import matmul
from timeit import default_timer as timer
from sklearn.cluster import DBSCAN, KMeans, HDBSCAN
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from placefm.utils import to_tensor, normalize_adj_tensor, estimate_eps

class POIPropagator:
    def __init__(self, args, alpha, beta, gamma, save_feats=False):
        self.args = args
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.save_feats = save_feats
        self.device = args.device


    def propagate(self, adj, feats):
        """

        Args:
            adj: The adjacency matrix.
            feats: The node features.

        Returns:
            The propagated features.
        """

        args = self.args
        start_total = timer()
        adj = to_tensor(adj, device=self.device)
        adj = normalize_adj_tensor(adj, sparse=True)
        
        feat_agg_hop0 = to_tensor(feats, device=self.device)
        feat_agg_hop1 = matmul(adj, feat_agg_hop0)
        feat_agg_hop2 = matmul(adj, feat_agg_hop1)

        propagated_feats = self.alpha * feat_agg_hop1 + self.beta * feat_agg_hop2 + self.gamma * feat_agg_hop0
        
        if self.args.verbose:
                end_agg = timer()
                args.logger.info(
                    f"=== Finished POI Propagation in {end_agg - start_total:.2f} sec ==="
                )

        if self.save_feats:
            # Save the propagated features
            save_path = f"../checkpoints/placefm_propagated_pois_{args.city}.pt"
            torch.save(propagated_feats, save_path)

        return propagated_feats


class Clusterer:
    def __init__(self, args, method):
        self.args = args
        self.method = method
        

    def cluster(self, feats, region_id, categories=None, lat_lon=None, edge_index=None, visualize=False):   
        """
        Clusters the features into regions.
        
        Args:
            feats: The features to cluster.
            region_id: The labels for the features (if applicable).
        
        Returns:
            The clustered regions.
        """
        
        if self.method == 'kmeans':
           
            n_clusters = max(1, int(feats.size(0) * self.args.placefm_kmeans_reduction_ratio))  # Ensure at least one cluster
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.args.seed)
            cluster_ids = kmeans.fit_predict(feats.cpu().numpy())
            unique_labels = set(cluster_ids)

            centroids = torch.tensor(kmeans.cluster_centers_, device=feats.device)
            cluster_sizes = torch.tensor(
                [sum(cluster_ids == label) for label in unique_labels if label != -1],
                device=feats.device
            )
        
        elif self.method == 'dbscan':
            
            # Estimate eps using the k-th nearest neighbor distance method and perform clustering
            eps = estimate_eps(feats.cpu().numpy(), min_samples=min_samples)
            # hdbscan = HDBSCAN(min_cluster_size=min_samples, metric='euclidean')
            dbscan = DBSCAN(min_samples=min_samples, eps=0.1, metric='euclidean')
            cluster_ids = dbscan.fit_predict(feats.cpu().numpy())
            unique_labels = set(cluster_ids)

            # Reassign noise points (-1) to the nearest cluster
            feats_np = feats.cpu().numpy()
            for idx, cluster_id in enumerate(cluster_ids):
                if cluster_id == -1:  # If the point is noise
                    distances = [np.linalg.norm(feats_np[idx] - feats_np[cluster_ids == unique_label].mean(axis=0)) 
                                 for unique_label in unique_labels if unique_label != -1]
                    if distances:  # If there are valid clusters
                        cluster_ids[idx] = np.argmin(distances)  # Assign to the nearest cluster

            # calculate centroids for each cluster
            centroids = []
            feats_np = feats.cpu().numpy()
            for label in unique_labels:
                cluster_points = feats_np[cluster_ids == label]
                centroid = cluster_points.mean(axis=0) if cluster_points.size > 0 else np.zeros(feats_np.shape[1])
                centroids.append(centroid)
            centroids = torch.tensor(np.array(centroids), device=feats.device)

        else:
            raise ValueError(f"Unknown clustering method: {self.method}")


        # visualization of clustering POIs
        if visualize:
            # Perform t-SNE for dimensionality reduction to 2D
            feats_np = feats.cpu().numpy()
            tsne = TSNE(n_components=2, random_state=self.args.seed)
            feats_2d = tsne.fit_transform(feats_np)

            # Plot the clustering results
            plt.figure(figsize=(10, 8))
            # feats_2d = feats_np[:, :2]  # Assuming the first two dimensions for 2D visualization

            # Plot clusters
            for label in unique_labels:
                cluster_points = feats_2d[cluster_ids == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', alpha=0.7)

            # Highlight centroids
            centroids_2d = np.array([feats_2d[cluster_ids == label].mean(axis=0) for label in unique_labels if label != -1])
            plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=100, label='Centroids')
            # Annotate each point with its category
            for i, (x, y) in enumerate(feats_2d):
                annotation = f"{i}: {str(categories[i]).split('>')[-1]}"
                plt.text(x, y, annotation, fontsize=8, ha='center', va='center')

            plt.title(f't-SNE Clustering Results for region {region_id}')
            plt.legend()
            plt.grid(True)

            # Save the figure
            save_path = f"../checkpoints/logs/placefm/clustering_results_region_{region_id}_in_{self.args.city}.png"
            plt.savefig(save_path)
            plt.close()


            # Plot the points in geographical space
            plt.figure(figsize=(25, 22))  # Increase the figure size to make the points more separated
            longitudes = lat_lon[:, 0]  # Longitude
            latitudes = lat_lon[:, 1]  # Latitude

            plt.scatter(longitudes, latitudes, alpha=0.7, label='Points')

            # Annotate each point with its category and index
            for i in range(len(lat_lon)):
                annotation = f"{i}: {str(categories[i]).split('>')[-1]}"
                plt.text(longitudes[i], latitudes[i], annotation, fontsize=8, ha='center', va='center')

            # Plot edges if edge_index is provided
            if edge_index is not None:
                for i, edge in enumerate(edge_index.T):
                    start, end = edge
                    plt.plot(
                        [longitudes[start], longitudes[end]],
                        [latitudes[start], latitudes[end]],
                        color='gray', alpha=1.0, linewidth=0.5
                    )

            plt.title(f'Geographical Clustering Results for region {region_id}')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()
            plt.grid(True)

            # Save the geographical plot
            geo_save_path = f"../checkpoints/logs/placefm/geographical_clustering_region_{region_id}_in_{self.args.city}.png"
            plt.savefig(geo_save_path)
            plt.close()


            return centroids
        
        
        return centroids, cluster_sizes


class Aggregator:
    def __init__(self, args, method):
        self.args = args
        self.method = method
    
    def aggregate(self, centroids, cluster_weights):
        """
        Aggregates the centroids of the region embedding to generate a single embedding for the region.

        Args:
            centroids: A tensor containing the centroids of the region.
            cluster_weight: A list containing the weights for each cluster.

        Returns:
            A single embedding representing the region.
        """
        if self.method == 'mean':
            # Compute the weighted mean of the centroids
            region_embedding = (centroids * cluster_weights.unsqueeze(1)).sum(dim=0)
        elif self.method == 'max':
            # Compute the max of the centroids
            region_embedding, _ = centroids.max(dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")

        return region_embedding