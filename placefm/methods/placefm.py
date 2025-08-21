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
        


    def generate_embeddings(self, verbose=False):
    
        args = self.args

        start_total = timer()
        # First, propagate POI features
        propagated_pois = self.propagator.propagate(self.data.adj_full, self.data.x)
        # Second, cluster the propagated POIs into places (regions)
        
        region_emb_list = []
        region_emb_ids = []

        # Assuming data.y contains class labels for each POI
        unique_classes = torch.unique(self.data.y)
        for cls in unique_classes:
            region_id = int(cls)
            idx = (self.data.y == region_id).nonzero(as_tuple=True)[0]

            region_pois = propagated_pois[idx]
            # region_categories = self.data.category.iloc[idx.tolist()].tolist()
            # region_lat_lon = self.data.lat_lon[idx.cpu()]

            # # Extract edges within the region
            # region_edge_index = []

            # poi_idx_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(idx)}
            # for edge_idx in range(self.data.edge_index.size(1)):
            #     src, dst = self.data.edge_index[:, edge_idx]
            #     if src in idx and dst in idx:
            #         region_edge_index.append([poi_idx_map[src.item()], poi_idx_map[dst.item()]])

            # region_edge_index = torch.tensor(region_edge_index, dtype=torch.long).t().contiguous()

            min_samples = 5
            if region_pois.size(0) <= min_samples:  # Check if the number of pois is less than or equal to min_samples
                region_emb = region_pois.mean(dim=0, keepdim=True)
            else:
                # Cluster POIs of this class
                centroids, cluster_sizes = self.clusterer.cluster(region_pois, region_id, visualize=False)
                # Scale cluster sizes to [0, 1] to use as weights
                cluster_sizes = cluster_sizes.float()
                cluster_weights = cluster_sizes / cluster_sizes.sum()

                # Aggregate cluster centroids to get region embeddings
                region_emb = self.aggregator.aggregate(centroids, cluster_weights)

            region_emb_list.append(region_emb)
            region_emb_ids.append(region_id)


        region_emb = torch.vstack(region_emb_list)
        import ipdb; ipdb.set_trace()

        end_total = timer()
        if verbose:
            args.logger.info(
                f"=== Finished generating trained region embeddings in {end_total - start_total:.2f} sec ==="
            )
        
        print(f"Total number of generated regions in {args.city}: {region_emb.size(0)}")
        saved_path = f"../checkpoints/placefm_{args.city}_region_embs.pt"

        
        # Save both region embeddings and region IDs in a dictionary

        save_obj = {
            "x": region_emb,
            "region_id": region_emb_ids
        }

        torch.save(save_obj, saved_path)
        print(f"Region embeddings of {args.city} has been save to {saved_path}")

        return save_obj