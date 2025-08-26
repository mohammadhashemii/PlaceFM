# https://github.com/google-research/population-dynamics

import os
import numpy as np
import pandas as pd
import torch
import pytorch_warmup as warmup
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from timeit import default_timer as timer
import math


class PDFM:
    """
    PDFM (Population Dynamics Foundation Model).
    It loads the pretrained region(zipcode/county) embeddings.
    """

    def __init__(self, data, args, **kwargs):
        self.data = data
        self.args = args
        self.device = args.device

        self.zcta_embeddings = pd.read_csv(os.path.join(args.load_path, f"pdfm_embeddings/v0/us/zcta_embeddings.csv"))

    def load_embeddings(self, features="all"):
        args = self.args

        
        # Extract the embeddings for the features and filter based on matching regions
        selected_embeddings = []
        unique_regions = self.data.y.unique().cpu().numpy()
        
        for region in unique_regions:
            # Format the region to match the "place" column format
            formatted_region = f"zip/{region}"
            matching_rows = self.zcta_embeddings[self.zcta_embeddings['place'] == formatted_region]
    
            if features == "all":
                if not matching_rows.empty:
                    feature_names = [f"feature{i}" for i in range(0, 330)]
                    selected_embeddings.append(matching_rows[feature_names].values)
                else:
                    selected_embeddings.append([0.0] * 330)  # Append zero vector if no match found
                    print(f"Warning: No matching embedding found for region {formatted_region}")
            
            elif features == "maps":
                if not matching_rows.empty:
                    feature_names = [f"feature{i}" for i in range(128, 256)]
                    selected_embeddings.append(matching_rows[feature_names].values)
                else:
                    selected_embeddings.append([0.0] * 128)
            



        region_emb = np.array(selected_embeddings)[:,0,:]
        region_emb_ids = unique_regions


        return region_emb, region_emb_ids

    def generate_embeddings(self, verbose=False, save_path=None):
        
        args = self.args

        start_total = timer()
        region_emb, region_emb_ids = self.load_embeddings(features="maps")
        end_total = timer()
        
        if verbose:
            args.logger.info(
                f"=== Finished generating trained region embeddings in {end_total - start_total:.2f} sec ==="
            )
        
        print(f"Total number of generated regions in {args.city}: {region_emb.shape[0]}")
        
        # Save both region embeddings and region IDs in a dictionary
    
        save_obj = {
            "x": region_emb,
            "region_id": region_emb_ids
        }

        if save_path is not None:
            save_path = f"../checkpoints/pdfm_{args.city}_region_embs.pt"
            torch.save(save_obj, save_path)
            print(f"Region embeddings of {args.city} has been save to {save_path}")

        return save_obj
