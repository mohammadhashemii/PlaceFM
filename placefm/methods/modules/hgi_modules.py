# https://github.com/RightBank/HGI/blob/main/Module/hgi_module.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import reset, uniform
from torch_geometric.nn import GCNConv
import random
import math

EPS = 1e-15

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
    
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class PMA(nn.Module):
    """the aggregation from POIs to regions function based on multi-head attention mechanisms"""
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
    

class POIEncoder(nn.Module):
    """POI GCN encoder"""
    def __init__(self, in_channels, hidden_channels):
        super(POIEncoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=True, bias=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)
        return x
    
class POI2Region(nn.Module):
    """POI - region aggregation and GCN at regional level"""
    def __init__(self, hidden_channels, num_heads):
        super(POI2Region, self).__init__()
        self.PMA = PMA(dim=hidden_channels, num_heads=num_heads, num_seeds=1, ln=False)
        self.conv = GCNConv(hidden_channels, hidden_channels, cached=True, bias=True)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x, zone, region_adjacency):
        region_ids = torch.unique(zone)
        region_emb = x.new_zeros((region_ids.size(0), x.size(1)))
        for idx, region_id in enumerate(region_ids):
            poi_indices = (zone == region_id).nonzero(as_tuple=True)[0]
            if poi_indices.numel() > 0:
                region_emb[idx] = self.PMA(x[poi_indices].unsqueeze(0)).squeeze(0)
        

        # Convert adjacency matrix to edge_index if needed
        if region_adjacency.dim() == 2 and region_adjacency.size(0) == region_adjacency.size(1):
            edge_index = region_adjacency.nonzero(as_tuple=False).t()
            edge_weight = region_adjacency[region_adjacency.nonzero(as_tuple=True)]
            region_emb = self.conv(region_emb, edge_index, edge_weight)
        else:
            region_emb = self.conv(region_emb, region_adjacency)
        
        region_emb = self.prelu(region_emb)
        return region_emb
    
def corruption(x):
    """corruption function to generate negative POIs through random permuting POI initial features"""
    return x[torch.randperm(x.size(0))]


class HierarchicalGraphInfomax(torch.nn.Module):
    r"""The Hierarchical Graph Infomax Module for learning region representations"""
    def __init__(self, hidden_channels, poi_encoder, poi2region, region2city, corruption, alpha):
        super(HierarchicalGraphInfomax, self).__init__()
        self.hidden_channels = hidden_channels
        self.poi_encoder = poi_encoder
        self.poi2region = poi2region
        self.region2city = region2city
        self.corruption = corruption
        self.alpha = alpha
        self.weight_poi2region = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.weight_region2city = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.region_embedding = torch.tensor(0)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.poi_encoder)
        reset(self.poi2region)
        reset(self.region2city)
        uniform(self.hidden_channels, self.weight_poi2region)
        uniform(self.hidden_channels, self.weight_region2city)

    def forward(self, data):
        """forward function to generate POI, region, and city representations"""
        pos_poi_emb = self.poi_encoder(data.x, data.edge_index, data.edge_weight)
        cor_x = self.corruption(data.x)
        neg_poi_emb = self.poi_encoder(cor_x, data.edge_index, data.edge_weight)
        # Generate region embeddings
        region_emb = self.poi2region(pos_poi_emb, data.y, data.region_adjacency)
        self.region_embedding = region_emb
        neg_region_emb = self.poi2region(neg_poi_emb, data.y, data.region_adjacency)
        city_emb = self.region2city(region_emb, data.region_area)

        pos_poi_emb_list = []
        neg_poi_emb_list = []
        """hard negative sampling procedure"""
        region_ids = torch.unique(data.y)


        ##### NOT SURE ABOUT THIS PART ########
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        
        for region_id in region_ids:
            id_of_poi_in_a_region = (data.y == region_id).nonzero(as_tuple=True)[0]
            poi_emb_of_a_region = pos_poi_emb[id_of_poi_in_a_region]
            hard_negative_choice = random.random()
            if hard_negative_choice < 0.25 and hasattr(data, 'coarse_region_similarity'):
                # Find similar regions based on coarse_region_similarity
                region_idx = (region_ids == region_id).nonzero(as_tuple=True)[0].item()
                similarity = data.coarse_region_similarity[region_idx]
                hard_example_range = ((similarity > 0.6) & (similarity < 0.8)).nonzero(as_tuple=True)[0]
                if hard_example_range.size(0) > 0:
                    another_region_idx = random.choice(hard_example_range.tolist())
                    another_region_id = region_ids[another_region_idx]
                else:
                    other_region_ids = region_ids[region_ids != region_id]
                    another_region_id = random.choice(other_region_ids.tolist())
            else:
                other_region_ids = region_ids[region_ids != region_id]
                another_region_id = random.choice(other_region_ids.tolist())
            id_of_poi_in_another_region = (data.y == another_region_id).nonzero(as_tuple=True)[0]
            poi_emb_of_another_region = pos_poi_emb[id_of_poi_in_another_region]
            pos_poi_emb_list.append(poi_emb_of_a_region)
            neg_poi_emb_list.append(poi_emb_of_another_region)
        return pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb

    def discriminate_poi2region(self, poi_emb_list, region_emb, sigmoid=True):
        values = []
        for region_id, region in enumerate(poi_emb_list):
            if region.size()[0] > 0:
                region_summary = region_emb[region_id]
                value = torch.matmul(region, torch.matmul(self.weight_poi2region, region_summary))
                values.append(value)
        values = torch.cat(values, dim=0)
        return torch.sigmoid(values) if sigmoid else values

    def discriminate_region2city(self, region_emb, city_emb, sigmoid=True):
        value = torch.matmul(region_emb, torch.matmul(self.weight_region2city, city_emb))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb):
        r"""Computes the mutual information maximization objective among the POI-region-city hierarchy."""
        pos_loss_region = -torch.log(
            self.discriminate_poi2region(pos_poi_emb_list, region_emb, sigmoid=True) + EPS).mean()
        neg_loss_region = -torch.log(
            1 - self.discriminate_poi2region(neg_poi_emb_list, region_emb, sigmoid=True) + EPS).mean()
        pos_loss_city = -torch.log(
            self.discriminate_region2city(region_emb, city_emb, sigmoid=True) + EPS).mean()
        neg_loss_city = -torch.log(
            1 - self.discriminate_region2city(neg_region_emb, city_emb, sigmoid=True) + EPS).mean()
        loss_poi2region = pos_loss_region + neg_loss_region
        loss_region2city = pos_loss_city + neg_loss_city
        return loss_poi2region * self.alpha + loss_region2city * (1 - self.alpha)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.hidden_channels)

    def get_region_emb(self):
        return self.region_embedding.clone().cpu().detach()