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

class Place2Vec:
    """
    Averaging (Averaging embeddings in each region).
    It outputs region embeddings.
    """

    def __init__(self, data, args, **kwargs):
        self.data = data
        self.args = args
        self.device = args.device

    def generate_embeddings(self, verbose=False, save_path=None):
    
        args = self.args
        total_time = 0
        
        poi_features = self.data.x
        
        region_emb_list = []
        region_emb_ids = []

        # Assuming data.y contains class labels for each POI
        unique_classes = torch.unique(self.data.y)
        for cls in unique_classes:
            region_id = int(cls)
            idx = (self.data.y == region_id).nonzero(as_tuple=True)[0]

            region_pois = poi_features[idx]
            region_emb = region_pois.mean(dim=0, keepdim=True)

            region_emb_list.append(region_emb)
            region_emb_ids.append(region_id)

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
            save_path = f"../checkpoints/place2vec/{args.state}_region_embs.pt"
            torch.save(save_obj, save_path)
            print(f"Region embeddings of {args.state} has been save to {save_path}")

        return save_obj