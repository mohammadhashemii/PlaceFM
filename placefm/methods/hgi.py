# https://github.com/RightBank/HGI/blob/main/train.py

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

from methods.modules.hgi_modules import *
from tqdm import tqdm

class HGI:
    """
    HGI (hierarchical Graph Infomax).
    It outputs region embeddings.
    """

    def __init__(self, data, args, **kwargs):
        self.data = data
        self.args = args
        self.device = args.device

        self.model = HierarchicalGraphInfomax(
            hidden_channels=data.x.shape[1],
            poi_encoder=POIEncoder(data.x.shape[1], data.x.shape[1]),
            poi2region=POI2Region(data.x.shape[1], args.hgi_attention_head),
            region2city=lambda z, area: torch.sigmoid((z.transpose(0, 1) * area).sum(dim=1)),
            corruption=corruption,
            alpha=args.hgi_alpha,
        ).to(self.device)


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=args.hgi_gamma)
        self.warmup_scheduler = warmup.LinearWarmup(self.optimizer, args.warmup_period)


    def train_epoch(self):
        args = self.args

        self.model.train()
        self.optimizer.zero_grad()

        pos_poi_emb_list, neg_poi_emb_list, region_emb, region_emb_ids, neg_region_emb, city_emb = self.model(self.data)
        loss = self.model.loss(pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=args.max_norm)

        self.optimizer.step()
        with self.warmup_scheduler.dampening():
            self.scheduler.step()
        return loss.item(), region_emb_ids

    def train(self):
        args = self.args
        
        print(f"Start training region embeddings for the city of {args.city}")
        lowest_loss = math.inf
        region_emb_to_save = torch.FloatTensor(0)
        for epoch in range(1, args.epochs + 1):
            loss, region_emb_ids = self.train_epoch()
            if epoch % 2 == 0 or epoch == 1 or epoch == args.epochs:
                args.logger.info(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f}")
            if loss < lowest_loss:
            # Save the embeddings with the lowest loss
                region_emb_to_save = self.model.get_region_emb()
                lowest_loss = loss

        print(f"Finished training region embeddings for {args.city} with lowest loss: {lowest_loss:.4f}")

        return region_emb_to_save, region_emb_ids
    

    def generate_embeddings(self, verbose=False, save_path=None):
        
        args = self.args

        start_total = timer()
        region_emb, region_emb_ids = self.train()
        end_total = timer()
        
        if verbose:
            args.logger.info(
                f"=== Finished generating trained region embeddings in {end_total - start_total:.2f} sec ==="
            )
        
        print(f"Total number of generated regions in {args.city}: {region_emb.size(0)}")
        
        # Save both region embeddings and region IDs in a dictionary
    
        save_obj = {
            "x": region_emb,
            "region_id": region_emb_ids
        }

        if save_path is not None:
            save_path = f"../checkpoints/hgi_{args.city}_region_embs.pt"
            torch.save(save_obj, save_path)
            print(f"Region embeddings of {args.city} has been save to {save_path}")

        return save_obj
