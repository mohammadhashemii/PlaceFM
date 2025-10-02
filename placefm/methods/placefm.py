import os
import torch
import pytorch_warmup as warmup
import numpy as np
from numpy import random
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from tqdm import trange
from timeit import default_timer as timer
import math

from methods.modules.placefm_modules import *
from tqdm import tqdm

class PlaceFM:
    """
    PlaceFM (Place Foundation Model).
    It outputs region embeddings.
    """

    def __init__(self, data, args, **kwargs):
        self.data = data
        self.args = args
        self.device = args.device

        self.propagator = POIPropagator(args, args.placefm_agg_alpha, args.placefm_agg_beta, args.placefm_agg_gamma, save_feats=False)
        self.clusterer = Clusterer(args, method=args.clustering_method)
        self.aggregator = Aggregator(args, method=args.region_agg_method)
        self.visualizer = Visualizer(args)


    def generate_embeddings(self, verbose=False, save_path=None):
    
        args = self.args
        total_time = 0
        
        # First, propagate POI features
        if args.no_prop:
            propagated_pois = self.data.x
        else:
            propagated_pois = self.propagator.propagate(self.data.adj_full, self.data.x)
        # Second, cluster the propagated POIs into places (regions)
        
        region_emb_list = []
        region_emb_ids = []

        # Assuming data.y contains class labels for each POI
        unique_classes = torch.unique(self.data.y)
        for cls in unique_classes:
            region_id = int(cls)
            if cls != 30329:
                continue
            idx = (self.data.y == region_id).nonzero(as_tuple=True)[0]

            region_pois = propagated_pois[idx]
            # region_pois = self.data.x[idx]


            min_samples = 5
            if region_pois.size(0) <= min_samples:  # Check if the number of pois is less than or equal to min_samples
                region_emb = region_pois.mean(dim=0, keepdim=True)
            else:
                # Cluster POIs of this class
                start_total = timer()
                centroids, cluster_sizes, cluster_ids = self.clusterer.cluster(region_pois)
                # Scale cluster sizes to [0, 1] to use as weights
                cluster_sizes = cluster_sizes.float()
                cluster_weights = cluster_sizes / cluster_sizes.sum()

                # Aggregate cluster centroids to get region embeddings
                region_emb = self.aggregator.aggregate(centroids, cluster_weights)
                end_total = timer()
                total_time += end_total - start_total
                # print(f"=== Time till now in {total_time:.2f} sec ===")


                # visualize region's identified places

                
                region_categories = self.data.category.iloc[idx.tolist()].tolist()
                region_lat_lon = self.data.lat_lon[idx.cpu()]         
                region_edge_index = []
                region_edge_weight = []

                poi_idx_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(idx)}
                # Mask for edges where both src and dst are in idx
                src_nodes = self.data.edge_index[0]
                dst_nodes = self.data.edge_index[1]
                edge_weights = self.data.edge_weight
                idx_set = set(idx.tolist())
                mask = torch.tensor([(src.item() in idx_set and dst.item() in idx_set) for src, dst in zip(src_nodes, dst_nodes)], dtype=torch.bool)

                # Filter edges and their weights
                filtered_src = src_nodes[mask]
                filtered_dst = dst_nodes[mask]
                filtered_weights = edge_weights[mask]

                # Map old indices to new indices
                mapped_src = torch.tensor([poi_idx_map[src.item()] for src in filtered_src], dtype=torch.long)
                mapped_dst = torch.tensor([poi_idx_map[dst.item()] for dst in filtered_dst], dtype=torch.long)

                region_edge_index = torch.stack([mapped_src, mapped_dst], dim=0)
                region_edge_weight = filtered_weights

                self.visualizer.visualize(region_pois, region_id, cluster_ids=cluster_ids, categories=region_categories, lat_lon=region_lat_lon, edge_index=region_edge_index, edge_weight=region_edge_weight)
    

            region_emb_list.append(region_emb)
            region_emb_ids.append(region_id)

            if verbose:
                args.logger.info(f"=== Finished clustering region {region_id} with {region_pois.size(0)} pois ===")

        region_emb = torch.vstack(region_emb_list)

        
        if verbose:
            args.logger.info(
                f"=== Finished generating trained region embeddings in {total_time:.2f} sec ==="
            )
        
        print(f"Total number of generated regions in {args.state}: {region_emb.size(0)}")

        
        # Save both region embeddings and region IDs in a dictionary

        save_obj = {
            "x": region_emb,
            "region_id": region_emb_ids
        }

        if save_path is not None:
            save_path = f"../checkpoints/placefm/{args.state}_region_embs.pt"
            torch.save(save_obj, save_path)
            print(f"Region embeddings of {args.state} has been save to {save_path}")

        return save_obj