import torch
from torch_sparse import matmul
from timeit import default_timer as timer
from sklearn.cluster import HDBSCAN
from sklearn.cluster import BisectingKMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
import numpy as np

from placefm.utils import to_tensor, normalize_adj_tensor, estimate_eps
from placefm.utils import plot_tsne_pois, plot_spatial_pois_folium, assign_cluster_colors

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
            save_path = f"../checkpoints/placefm_propagated_pois_{args.state}.pt"
            torch.save(propagated_feats, save_path)

        return propagated_feats


class Clusterer:
    def __init__(self, args, method):
        self.args = args
        self.method = method
        

    def cluster(self, feats):   
        """
        Clusters the features into regions.
        
        Args:
            feats: The features to cluster.
        
        Returns:
            The clustered regions.
        """
        if self.method == "agg":
            n_clusters = max(1, int(feats.size(0) * self.args.placefm_kmeans_reduction_ratio))  # Ensure at least one cluster
            agg = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_ids = agg.fit_predict(feats.cpu().numpy())
            unique_labels = set(cluster_ids)
            print(f"Number of regions found by Agglomerative Clustering: {len(unique_labels)}")

            # calculate centroids for each cluster
            centroids = []
            feats_np = feats.cpu().numpy()
            for label in unique_labels:
                cluster_points = feats_np[cluster_ids == label]
                centroid = cluster_points.mean(axis=0) if cluster_points.size > 0 else np.zeros(feats_np.shape[1])
                centroids.append(centroid)
            centroids = torch.tensor(np.array(centroids), device=feats.device)
            cluster_sizes = torch.tensor(
                [sum(cluster_ids == label) for label in unique_labels if label != -1],
                device=feats.device
            )

        elif self.method == "affinity":
            affinity = 'euclidean'  # 'precomputed' or 'euclidean'
            damping = 0.95  # between 0.5 and 1
            preference = None  # if None, the median of the input similarities is used
            max_iter = 200
            convergence_iter = 15

            af = AffinityPropagation(affinity=affinity, damping=damping, 
                                    preference=preference, max_iter=max_iter, 
                                    convergence_iter=convergence_iter)
            cluster_ids = af.fit_predict(feats.cpu().numpy())
            unique_labels = set(cluster_ids)
            if self.args.verbose:
                print(f"Number of places found by Affinity Propagation: {len(unique_labels)}")

            centroids = torch.tensor(af.cluster_centers_, device=feats.device)
            cluster_sizes = torch.tensor(
                [sum(cluster_ids == label) for label in unique_labels if label != -1],
                device=feats.device
            )
            
        elif self.method == 'kmeans':
            
            n_clusters = max(1, int(feats.size(0) * self.args.placefm_kmeans_reduction_ratio))  # Ensure at least one cluster
            kmeans = BisectingKMeans(n_clusters=n_clusters, random_state=self.args.seed)
            cluster_ids = kmeans.fit_predict(feats.cpu().numpy())
            unique_labels = set(cluster_ids)

            centroids = torch.tensor(kmeans.cluster_centers_, device=feats.device)
            cluster_sizes = torch.tensor(
                [sum(cluster_ids == label) for label in unique_labels],
                device=feats.device
            )
            if cluster_sizes.shape[0] != centroids.shape[0]:
                centroids = centroids[:-1, :]
        
        elif self.method == 'dbscan':
            min_samples = 5
            hdbscan = HDBSCAN(min_cluster_size=min_samples, metric='euclidean',)
            cluster_ids = hdbscan.fit_predict(feats.cpu().numpy())
            unique_labels = set(cluster_ids)
            print(f"Number of regions found by HDBSCAN: {len(unique_labels)}")

            # Reassign noise points (-1) to the nearest cluster
            # feats_np = feats.cpu().numpy()
            # for idx, cluster_id in enumerate(cluster_ids):
            #     if cluster_id == -1:  # If the point is noise
            #         distances = [np.linalg.norm(feats_np[idx] - feats_np[cluster_ids == unique_label].mean(axis=0)) 
            #                      for unique_label in unique_labels if unique_label != -1]
            #         if distances:  # If there are valid clusters
            #             cluster_ids[idx] = np.argmin(distances)  # Assign to the nearest cluster

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

        return centroids, cluster_sizes, cluster_ids


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
    

class Visualizer:
    def __init__(self, args):
        self.args = args

    def visualize(self, region_pois, region_id, cluster_ids=None, categories=None, lat_lon=None, edge_index=None, edge_weight=None):
        """
        Visualizes the POIs in the region.

        Args:
            region_pois: The POIs in the region.
            region_id: The ID of the region.
        """
        args = self.args
        if args.verbose:
            print(f"Visualizing POIs in region {region_id} with {region_pois.size(0)} POIs...")
        
        # Example usage:
        cluster_color_map = assign_cluster_colors(region_pois, categories, cluster_ids, num_colors=len(set(cluster_ids)), method='random')


        # Perform t-SNE to show the clustering results in 2D space
        # plot_tsne_pois(self.args, region_pois, categories, cluster_ids, region_id)
        
        # Plot the POIs in geographical space
        # plot_spatial_pois(self.args, lat_lon, categories, edge_index, cluster_ids, region_id)
        plot_spatial_pois_folium(self.args, lat_lon, categories, edge_index, edge_weight, cluster_ids, region_id, cluster_color_map)

        import ipdb; ipdb.set_trace()