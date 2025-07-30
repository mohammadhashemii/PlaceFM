import re
import os.path as osp
from typing import Callable, Optional

import numpy as np
import pandas as pd


import torch
from torch_geometric.data import InMemoryDataset, Data


from placefm.dataset.utils import *
import geopandas as gpd


class F_OSM(InMemoryDataset):
    r"""
    POI dataset for geolocation learning from fused Foursquare and OSM data.
    Each node represents a point-of-interest (POI) with geolocation features.

    Args:
        root (str): Root directory where the dataset should be stored.
        transform (callable, optional): A function/transform that takes in a
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
        pre_transform (callable, optional): A function/transform that takes in a
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before being saved to disk.
        force_reload (bool, optional): If True, will process the dataset again.
    """

    def __init__(
        self,
        root: str,
        name: str,
        args,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        self.args = args
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        
        
        self.load(self.processed_paths[0])
        

    

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return 'us_pois_clean.csv'

    @property
    def processed_file_names(self) -> str:
        return f'{self.args.city}.pt'

    def download(self) -> None:
        # Assume manual placement of raw CSV file
        pass

    

    def process(self) -> None:
        args = self.args
        poi_raw_feats = extract_feats(args=args, csv_path=self.raw_paths[0])

        assert len(poi_raw_feats) > 0, f"No POI records found within '{args.city}'. Please check the city name and try again."

        print(" 1. Extracting category embeddings...")
        emb = encode_category_levels(poi_raw_feats, level=6, method="SD-CEM")
        x = torch.tensor(emb.values, dtype=torch.float)

        # POI edge Creation
        lat = torch.tensor(poi_raw_feats['latitude'].values, dtype=torch.float).view(-1, 1)
        lon = torch.tensor(poi_raw_feats['longitude'].values, dtype=torch.float).view(-1, 1)
        lat_lon = torch.cat([lat, lon], dim=1)

        postcodes = poi_raw_feats['postcode'].tolist()
        y = postcodes

        print(f"2. Creating edges between POIs based on geographical distances using {args.edge_creation} aklgorithm...")
        edge_index, edge_weight = create_poi_edges(coords=lat_lon, method=args.edge_creation, region_labels=y)

        # Calculate region area for each unique postcode in the city using the shapefile
        print("2b. Calculating region areas...")
        unique_postcodes = list(set(y))
        region_areas = calculate_region_areas(unique_postcodes, shapefile_path='/scratch/mhashe4/repos/placefm/data/f-osm/Census/tl_2024_us_zcta520/tl_2024_us_zcta520.shp')
        data_region_area = torch.tensor([region_areas.get(str(postcode), 0.0) for postcode in unique_postcodes], dtype=torch.float)


        # Create region adjacency matrix: two regions are connected if their polygons touch
        print("3. Creating region adjacency matrix...")
        region_adj = create_region_adjacency(y=y, shapefile_path=f'/scratch/mhashe4/repos/placefm/data/f-osm/Census/tl_2024_us_zcta520/tl_2024_us_zcta520.shp')
        

        # All nodes are training nodes
        num_nodes = x.size(0)
        train_mask = torch.ones(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        idx_train = torch.arange(num_nodes)
        idx_val = torch.tensor([], dtype=torch.long)
        idx_test = torch.tensor([], dtype=torch.long)


        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y, region_adjacency=region_adj, region_area=data_region_area,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                    idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

        # meta data
        data.lat_lon = lat_lon

        category_levels = [f"category_lvl{i+1}" for i in range(6)]
        data.category = poi_raw_feats[category_levels] \
            .apply(lambda row: " > ".join([str(x) for x in row if pd.notnull(x)]), axis=1)

        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
