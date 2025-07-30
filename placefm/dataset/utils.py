import os
import re
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer

from placefm.dataset.convertor import *
from placefm.utils import is_sparse_tensor

def sparsify(model_type, adj_syn, args, verbose=False):
    """
    Applies sparsification to the adjacency matrix based on the model type and given arguments.

    This function modifies the adjacency matrix to make it sparser according to the model type and method specified.
    For specific methods and datasets, it adjusts the threshold used for sparsification.

    Parameters
    ----------
    model_type : str
        The type of model used, which determines the sparsification strategy. Can be 'MLP', 'GAT', or other.
    adj_syn : torch.Tensor
        The adjacency matrix to be sparsified.
    args : argparse.Namespace
        Command-line arguments and configuration parameters which may include method-specific settings.
    verbose : bool, optional
        If True, prints information about the sparsity of the adjacency matrix before and after sparsification.
        Default is False.

    Returns
    -------
    adj_syn : torch.Tensor
        The sparsified adjacency matrix.
    """
    threshold = 0
    if model_type == "MLP":
        adj_syn = adj_syn - adj_syn
        torch.diagonal(adj_syn).fill_(1)
    elif model_type == "GAT":
        if args.method in ["gcond", "doscond"]:
            if args.dataset in ["cora", "citeseer"]:
                threshold = 0.5  # Make the graph sparser as GAT does not work well on dense graph
            else:
                threshold = 0.1
        elif args.method in ["msgc"]:
            threshold = args.threshold
        else:
            threshold = 0.5
    else:
        if args.method in ["gcond", "doscond"]:
            threshold = args.threshold
        elif args.method in ["msgc"]:
            threshold = 0
        else:
            threshold = 0

    # if verbose and args.method not in ['gcondx', 'doscondx', 'sfgc', 'geom', 'gcsntk']:
    # print('Sum:', adj_syn.sum().item())
    # print(adj_syn)
    # print('Sparsity:', adj_syn.nonzero().shape[0] / adj_syn.numel())
    # if args.method in ['sgdd']:
    #     threshold = 0.5
    if threshold > 0:
        adj_syn[adj_syn < threshold] = 0
        if verbose:
            print(
                "Sparsity after truncating:",
                adj_syn.nonzero().shape[0] / adj_syn.numel(),
            )
        # else:
        #     print("structure free methods do not need to truncate the adjacency matrix")
    return adj_syn


def index2mask(index, size):
    """
    Convert an index list to a boolean mask.

    Parameters
    ----------
    index : list or tensor
        List or tensor of indices to be set to True.
    size : int or tuple of int
        Shape of the mask. If an integer, the mask is 1-dimensional.

    Returns
    -------
    mask : tensor
        A boolean tensor of the specified size, with True at the given `index` positions and False elsewhere.

    Examples
    --------
    >>> index = [0, 2, 4]
    >>> size = 5
    >>> index2mask(index, size)
    tensor([True, False, True, False, True], dtype=torch.bool)
    """
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def splits(data, exp="default"):
    # customize your split here
    if hasattr(data, "y"):
        num_classes = max(data.y) + 1
    else:
        num_classes = max(data.labels_full).item() + 1
    # data.nclass = num_classes
    if not hasattr(data, "train_mask"):
        indices = []
        for i in range(num_classes):
            data.y = data.y.reshape(-1)
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        if exp == "random":
            train_index = torch.cat([i[:20] for i in indices], dim=0)
            val_index = torch.cat([i[20:50] for i in indices], dim=0)
            test_index = torch.cat([i[50:] for i in indices], dim=0)
        elif exp == "few":
            train_index = torch.cat([i[:5] for i in indices], dim=0)
            val_index = torch.cat([i[5:10] for i in indices], dim=0)
            test_index = torch.cat([i[10:] for i in indices], dim=0)
        else:
            # if fixed but no split is provided, use the default 8/1/1 split classwise
            train_index = torch.cat(
                [i[: int(i.shape[0] * 0.8)] for i in indices], dim=0
            )
            val_index = torch.cat(
                [i[int(i.shape[0] * 0.8) : int(i.shape[0] * 0.9)] for i in indices],
                dim=0,
            )
            test_index = torch.cat([i[int(i.shape[0] * 0.9) :] for i in indices], dim=0)
            # raise NotImplementedError('Unknown split type')
        data.train_mask = index2mask(train_index, size=data.num_nodes)
        data.val_mask = index2mask(val_index, size=data.num_nodes)
        data.test_mask = index2mask(test_index, size=data.num_nodes)
    data.idx_train = data.train_mask.nonzero().view(-1)
    data.idx_val = data.val_mask.nonzero().view(-1)
    data.idx_test = data.test_mask.nonzero().view(-1)

    return data


def save_reduced(adj_syn=None, feat_syn=None, labels_syn=None, args=None, method=None):
    if method is None:
        method = args.method
    save_path = f"{args.save_path}/reduced_graph/{args.method}"
    if args.attack is not None and args.dataset in ["flickr"]:
        save_path = (
            f"{args.save_path}/corrupt_graph/{args.attack}/reduced_graph/{args.method}"
        )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if adj_syn is not None:
        torch.save(
            adj_syn,
            f"{save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt",
        )
    if feat_syn is not None:
        torch.save(
            feat_syn,
            f"{save_path}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt",
        )
    if labels_syn is not None:
        torch.save(
            labels_syn,
            f"{save_path}/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt",
        )
    args.logger.info(
        f"Saved {save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt"
    )


def load_reduced(args, data=None):
    flag = 0
    save_path = f"{args.save_path}/reduced_graph/{args.method}"
    if args.attack is not None and args.dataset in ["flickr"]:
        save_path = (
            f"{args.save_path}/corrupt_graph/{args.attack}/reduced_graph/{args.method}"
        )
    try:
        feat_syn = torch.load(
            f"{save_path}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt",
            map_location=args.device,
            weights_only=True
        )
    except:
        print("find no feat, use original feature matrix instead")
        flag += 1
        if args.setting == "trans":
            feat_syn = data.feat_full
        else:
            feat_syn = data.feat_train
    try:
        labels_syn = torch.load(
            f"{save_path}/label_{args.dataset}_{args.reduction_rate}_{args.seed}.pt",
            map_location=args.device,
            weights_only=True
        )
    except:
        print("find no label, use original label matrix instead")
        flag += 1
        labels_syn = data.labels_train

    try:
        adj_syn = torch.load(
            f"{save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt",
            map_location=args.device,
            weights_only=True
        )
    except:
        print("find no adj, use identity matrix instead")
        flag += 1
        adj_syn = torch.eye(feat_syn.size(0), device=args.device)
    # args.logger.info(f"Load {save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt")
    assert flag < 3, "no file found, please run the reduction method first"

    return adj_syn, feat_syn, labels_syn


def get_syn_data(data, args, model_type, verbose=False):
    """
    Loads or computes synthetic data for evaluation.

    Parameters
    ----------
    data : Dataset
        The dataset containing the graph data.
    model_type : str
        The type of model used for generating synthetic data.
    verbose : bool, optional, default=False
        Whether to print detailed logs.

    Returns
    -------
    feat_syn : torch.Tensor
        Synthetic feature matrix.
    adj_syn : torch.Tensor
        Synthetic adjacency matrix.
    labels_syn : torch.Tensor
        Synthetic labels.
    """
    adj_syn, feat_syn, labels_syn = load_reduced(args, data)

    if labels_syn.shape[0] == data.labels_train.shape[0]:
        return feat_syn, adj_syn, labels_syn

    if type(adj_syn) == torch.tensor and is_sparse_tensor(adj_syn):
        adj_syn = adj_syn.to_dense()
    elif isinstance(adj_syn, torch.sparse.FloatTensor):
        adj_syn = adj_syn.to_dense()
    else:
        adj_syn = adj_syn

    adj_syn = sparsify(model_type, adj_syn, args, verbose=verbose)
    return feat_syn, adj_syn, labels_syn


# ---------------  F-OSM data loader utility functions  ------------------
def calculate_region_areas(unique_postcodes, shapefile_path):
    """
    Calculate the area (in square kilometers) for each region (ZIP code) in unique_postcodes using the provided shapefile.

    Args:
        unique_postcodes (list or array-like): List of ZIP codes to calculate areas for.
        shapefile_path (str): Path to the ZIP code shapefile.

    Returns:
        dict: Mapping from postcode (str) to area (float, in sq km).
    """

    # Load shapefile and ensure ZIP codes are strings
    zip_gdf = gpd.read_file(shapefile_path)
    zip_gdf['ZCTA5CE20'] = zip_gdf['ZCTA5CE20'].astype(str)

    # Filter to only relevant ZIP codes
    postcodes_str = [str(pc) for pc in unique_postcodes]
    filtered_gdf = zip_gdf[zip_gdf['ZCTA5CE20'].isin(postcodes_str)].copy()

    # Project to an equal-area CRS for accurate area calculation
    filtered_gdf = filtered_gdf.to_crs("EPSG:3395")  # World Mercator (meters)

    # Calculate area in square kilometers
    filtered_gdf['area_sqkm'] = filtered_gdf.geometry.area / 1e6

    # Build mapping
    area_dict = dict(zip(filtered_gdf['ZCTA5CE20'], filtered_gdf['area_sqkm']))

    # Normalize areas so that their sum is 1 (all non-None)
    total_area = sum(a for a in area_dict.values() if a is not None)
    if total_area > 0:
        for pc in area_dict:
            if area_dict[pc] is not None:
                area_dict[pc] = area_dict[pc] / total_area

    # Sanity checks: print largest and smallest area
    if area_dict:
        areas = [a for a in area_dict.values() if a is not None]
        if areas:
            max_area = max(areas)
            min_area = min(areas)
            max_zip = [pc for pc, a in area_dict.items() if a == max_area]
            min_zip = [pc for pc, a in area_dict.items() if a == min_area]
            print(f"Largest area: {max_area:.6f} (ZIPs: {max_zip})")
            print(f"Smallest area: {min_area:.6f} (ZIPs: {min_zip})")
        else:
            print("No valid areas found for sanity check.")
    else:
        print("Area dictionary is empty.")

    # For ZIPs not found, set area to None
    for pc in postcodes_str:
        if pc not in area_dict:
            area_dict[pc] = None

    return area_dict

def create_region_adjacency(y, shapefile_path):
    """
    Create adjacency matrix for regions (ZIP codes) based on polygon touching and proximity.

    Args:
        y (list or array-like): List of region labels (ZIP codes) present in POIs.
        shapefile_path (str): Path to the ZIP code shapefile.

    Returns:
        torch.Tensor: Region adjacency matrix of shape (num_regions, num_regions).
    """

    # Load ZIP code polygons
    zip_gdf = gpd.read_file(shapefile_path).to_crs("EPSG:3395")

    # Filter only regions present in POIs
    region_labels = list(set(y))
    # Ensure both are strings for matching
    zip_gdf['ZCTA5CE20'] = zip_gdf['ZCTA5CE20'].astype(str)
    region_labels_str = [str(r) for r in region_labels]
    zip_gdf = zip_gdf[zip_gdf['ZCTA5CE20'].isin(region_labels_str)].reset_index(drop=True)
    region_to_idx = {region: idx for idx, region in enumerate(region_labels)}
    num_regions = len(region_labels)
    region_adj = torch.zeros((num_regions, num_regions), dtype=torch.float)

    # Build adjacency: two regions are connected if their polygons touch
    for i, region_i in enumerate(sorted(region_labels)):
        poly_i_row = zip_gdf[zip_gdf['ZCTA5CE20'] == str(region_i)]
        if poly_i_row.shape[0] == 0:
            continue
        poly_i = poly_i_row.geometry.iloc[0]
        for j, region_j in enumerate(sorted(region_labels)):
            if i == j:
                continue
            poly_j_row = zip_gdf[zip_gdf['ZCTA5CE20'] == str(region_j)]
            if poly_j_row.shape[0] == 0:
                continue
            poly_j = poly_j_row.geometry.iloc[0]
            if poly_i.touches(poly_j):
                region_adj[i, j] = 1
                region_adj[j, i] = 1  # undirected
                print(f"Connected {region_i} <-> {region_j}")

    # If some regions have no neighbors, connect them to the closest region
    isolated = (region_adj.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    centroids = zip_gdf.geometry.centroid
    for idx in isolated.tolist():
        centroid_i = centroids.iloc[idx]
        min_dist = float('inf')
        min_j = None
        for j in range(num_regions):
            if j == idx:
                continue
            centroid_j = centroids.iloc[j]
            dist = centroid_i.distance(centroid_j)
            if dist < min_dist:
                min_dist = dist
                min_j = j
        if min_j is not None:
            region_adj[idx, min_j] = 1
            region_adj[min_j, idx] = 1  # undirected


    # Sanity check: ensure regions with common border are connected
    for i, region_i in enumerate(region_labels):
        poly_i = zip_gdf[zip_gdf['ZCTA5CE20'] == region_i].geometry.values
        if len(poly_i) == 0:
            continue
        poly_i = poly_i[0]
        for j, region_j in enumerate(region_labels):
            if i == j:
                continue
            poly_j = zip_gdf[zip_gdf['ZCTA5CE20'] == region_j].geometry.values
            if len(poly_j) == 0:
                continue
            poly_j = poly_j[0]
            if poly_i.touches(poly_j):
                if region_adj[i, j] != 1:
                    print(f"Sanity check failed: {region_i} and {region_j} share a border but are not connected in adjacency matrix.")


    return region_adj
        

def create_poi_edges(coords, method="knn", k=5, region_labels=None):
    """
    Create edges between POIs based on the selected method.

    Args:
        coords (np.ndarray): Array of shape (N, 2) with latitude and longitude.
        method (str): "knn" or "delaunay".
        k (int): Number of neighbors for KNN (ignored if method="delaunay").
        region_labels (torch.Tensor): shape of [num_nodes] (e.g., postcode labels)

    Returns:
        torch.LongTensor: Edge index tensor of shape (2, num_edges).
    """
    if method == "knn":
        coords_rad = np.radians(coords)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', metric='haversine')
        nbrs.fit(coords_rad)
        _, indices = nbrs.kneighbors(coords_rad)

        edge_index = []
        for src_idx, neighbors in enumerate(indices):
            for dst_idx in neighbors[1:]:  # skip self
                edge_index.append((src_idx, dst_idx))

        # Convert to tensor and make undirected
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    elif method == "dt":
        coords = np.array(coords)
        tri = Delaunay(coords)
        
        # Extract undirected edges from triangles
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                edges.add(tuple(sorted((a, b))))
        
        edge_index = []
        edge_weights = []

        # Compute L = diag length of bounding box
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        L = np.linalg.norm(max_coords - min_coords)  # Diagonal length

        for i, j in edges:
            # Compute spatial distance (Euclidean here; you can use haversine if needed)
            l_ij = np.linalg.norm(coords[i] - coords[j])

            # Region similarity weight
            wr = 1.0 if region_labels[i] == region_labels[j] else 0.4

            # Raw edge weight before normalization
            weight = np.log((1 + L ** 1.5) / (1 + l_ij ** 1.5)) * wr

            edge_index.append([i, j])
            edge_index.append([j, i])  # Undirected graph

            edge_weights.append(weight)
            edge_weights.append(weight)

        edge_index = torch.tensor(edge_index, dtype=torch.long).T  # shape [2, num_edges]
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)

        # Normalize edge weights to [0, 1]
        edge_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())

    else:
        raise ValueError(f"Unknown method: {method}. Choose 'knn' or 'dt'.")

    

    return edge_index, edge_weights


def extract_5digit_zip(z):
    if isinstance(z, str):
        match = re.match(r"^(\d{4,5})", z)
        return match.group(1) if match else None
    return None


def extract_feats(args, csv_path):
    df = pd.read_csv(csv_path)

    # 1. extract fsq_latitude and fsq_longitude
    features = df[["fsq_latitude", "fsq_longitude"]].rename(columns={"fsq_latitude": "latitude", "fsq_longitude": "longitude"})

    # 2. extract postcode
    features["postcode"] = df["fsq_postcode"].combine_first(df["osm_postcode"]) # there still will be some nan values for this column

    # 3. extract locality (city)
    features["locality"] = df["fsq_locality"].combine_first(df["osm_address"].str.extract(r'"city"\s*=>\s*"([^"]+)"')[0]) # there still will be some nan values for this column

    # 4. extract state
    features ["state"] = df["abbr"]

    # 5. extract fsq_date_created and fsq_date_closed
    features["date_created"] = df["fsq_date_created"]
    features["date_closed"] = df["fsq_date_closed"]

    # 6. 
    categories_df = pd.read_csv("/scratch/mhashe4/repos/fm/data/f-osm/raw/personalization-apis-movement-sdk-categories.csv")
    category_id_to_label = dict(zip(categories_df["Category ID"], categories_df["Category Label"]))


    def parse_category_ids(x):
        if pd.isna(x):
            return []
        cleaned = re.sub(r"[\'\[\]]", "", str(x)).strip()
        return cleaned.split()

    # Step 2: Dynamically extract all category levels
    def get_category_levels_dynamic(id_list):
        level_lists = []

        if not isinstance(id_list, list):
            return []

        for cid in id_list:
            label = category_id_to_label.get(cid)
            if label:
                parts = label.replace('"', '').replace("'", '').replace('[', '').replace(']', '').strip().split(" > ")
                
                # Ensure we have a list of lists where each sublist holds one level across all rows
                for i, part in enumerate(parts):
                    if len(level_lists) <= i:
                        level_lists.append([])
                    level_lists[i].append(part)

        return level_lists
    
    # Step 3: Apply parsing to the DataFrame
    df["fsq_category_ids"] = df["fsq_category_ids"].apply(parse_category_ids)

    # Extract category levels dynamically
    category_levels = df["fsq_category_ids"].apply(get_category_levels_dynamic)

    # Step 4: Normalize into columns
    # Determine max number of levels across all rows
    max_levels = category_levels.apply(len).max()

    # Initialize empty columns in features
    for i in range(max_levels):
        level_col_name = f"category_lvl{i+1}"
        features[level_col_name] = category_levels.apply(lambda x: x[i][-1] if len(x) > i and isinstance(x[i], list) and x[i] else None)

    features = features[features["category_lvl1"].notna()].reset_index(drop=True)
    features = features[features["locality"].notna()].reset_index(drop=True)

    features["postcode"] = features["postcode"].apply(extract_5digit_zip)
    features = features[features["postcode"].notna()].reset_index(drop=True)
    features = features[features["postcode"].astype(str).str.match(r"^\d{4,5}$")] # filter out those which are not 4 or 5 digit zipcodes
    
    # match with pdfm embeddings
    pdfm_zipcode_embeddings = pd.read_csv("/scratch/mhashe4/repos/fm/pdfm/data/pdfm_embeddings/v0/us/zcta_embeddings.csv")
    pdfm_zipcode_embeddings["place"] = pdfm_zipcode_embeddings["place"].str.replace("zip/", "", regex=False)


    merged = features.merge(
        pdfm_zipcode_embeddings[["place", "state"]].rename(columns={"place": "postcode", "state": "state_embed"}),
        on="postcode",
        how="left"
    )

    # Keep only rows where state matches
    features = merged[merged["state"] == merged["state_embed"]].drop(columns=["state_embed"])


    # filter pois belonging to postcodes that are rare
    postcode_counts = features.groupby("postcode")["postcode"].transform("count")
    features = features[postcode_counts >= 10]

                
    features["postcode"] = features["postcode"].astype(int)
    features = features[features["locality"].str.lower() == args.city.lower()]


    return features


def encode_category_levels(categories_df, level: int, method="one-hot") -> pd.DataFrame:

    assert level >= 1, "Level must be >= 1"
    levels = [f"category_lvl{i+1}" for i in range(level)]

    if method == "one-hot":
        
        level_encodings = []

        for level in levels:
            col = categories_df[level]
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            encoded = encoder.fit_transform(col.values.reshape(-1, 1))

            encoded_df = pd.DataFrame(
                encoded,
                columns=[f"{level}_{cat}" for cat in encoder.categories_[0]],
                index=categories_df.index
            )
            level_encodings.append(encoded_df)

        x_df = pd.concat(level_encodings, axis=1)
        return x_df
    
    elif method == "SD-CEM":

        sd_cem_path = "/scratch/mhashe4/repos/fm/data/f-osm/raw/SD-CEM/SD-CEM#US#30.csv"
        cem_df = pd.read_csv(sd_cem_path)

        # Clean CEM label column
        cem_df.iloc[:, 0] = cem_df.iloc[:, 0].astype(str).str.replace("&", "and")
        cem_labels = cem_df.iloc[:, 0].tolist()

        # SBERT setup
        SBERT = SentenceTransformer("all-MiniLM-L6-v2")
        cem_label_embeddings = SBERT.encode(cem_labels, normalize_embeddings=True, show_progress_bar=False)

        # Label → vector mapping
        label_to_vec = {
            row[0]: row[1:].values.astype(float)
            for _, row in cem_df.iterrows()
        }

        embedding_dim = len(next(iter(label_to_vec.values())))
        default_embedding = [None] * embedding_dim

        # === STEP 1: Precompute all unique category labels
        all_labels = (
            categories_df[[f"category_lvl{i+1}" for i in range(level)]]
            .stack()
            .dropna()
            .astype(str)
            .str.replace("&", "and")
            .unique()
        )

        # === STEP 2: Encode all unique labels at once
        all_label_embeddings = SBERT.encode(all_labels, normalize_embeddings=True, show_progress_bar=False)

        # === STEP 3: Map label → SBERT embedding
        label_to_embedding = {
            label: emb for label, emb in zip(all_labels, all_label_embeddings)
        }

        # === STEP 4: Matching function per row
        def match(row):


            labels = [row.get(f"category_lvl{i+1}") for i in reversed(range(level))]
            labels = [str(lbl).replace("&", "and") for lbl in labels if isinstance(lbl, str) and lbl.replace("&", "and") in label_to_embedding]

            if not labels:
                return default_embedding

            # Get embeddings from precomputed dict
            label_embs = np.stack([label_to_embedding[lbl] for lbl in labels])

            # Cosine similarity with all CEM labels
            sim_matrix = cosine_similarity(label_embs, cem_label_embeddings)
            sim_matrix[sim_matrix < 0.6] = -1

            best_cem_indices = np.argmax(sim_matrix, axis=1)
            best_scores = np.max(sim_matrix, axis=1)

            valid_mask = best_scores > 0
            if np.any(valid_mask):
                first_valid_idx = np.argmax(valid_mask)
                cem_idx = best_cem_indices[first_valid_idx]
                return label_to_vec[cem_labels[cem_idx]]

        
            reversed_levels = [f"category_lvl{i+1}" for i in reversed(range(level))]
            for lvl in reversed_levels:
                label = row.get(lvl)
                if isinstance(label, str) and "government" in label.lower():
                    return label_to_vec.get("Government Building", default_embedding)
            


            return default_embedding

        # === STEP 5: Apply matching
        embeddings = categories_df.apply(match, axis=1, result_type='expand')
        embeddings.columns = [f"sd_cem_{i}" for i in range(embedding_dim)]

        return embeddings



# ---------------------------------------------------------------------




from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

import torch
import pyfpgrowth
import geopandas as gpd


def disjointed_union(tree_list, class_=None, device=None):
    r"""Computes disjointed union of trees inside tree_list. trees are in
    torch_geometric.data.Data format. Returns a torch_geometric.data.Data
    graph without the roots_to_embed information."""
    tree_list = [
        tree for tree in tree_list if tree.x.shape[0] > 1
    ]  # skipping single node trees for now because it causes issues
    total_nodes = sum(tree.x.shape[0] for tree in tree_list)
    total_edges = sum(tree.edge_index.shape[1] for tree in tree_list)
    if total_nodes <= 1:
        return None  # if the whole graph has just one node, then we return none
    if device is None:
        device = tree_list[0].x.device
    assert all(
        device == tree.x.device for tree in tree_list
    ), "Trees must be on same device."
    assert all(
        tree.x.shape[1] == tree_list[0].x.shape[1] for tree in tree_list
    ), "Number of node features are different."
    assert all(
        tree.y == tree_list[0].y for tree in tree_list
    ), "Tree must be in same class."
    if hasattr(tree_list[0], "edge_attr"):
        assert all(
            hasattr(tree, "edge_attr") for tree in tree_list
        ), "Not all trees have edge features."
        assert all(
            tree.edge_attr.shape[1] == tree_list[0].edge_attr.shape[1]
            for tree in tree_list
        ), "Number of edge features are different."
        hasedgeattr = True
    else:
        hasedgeattr = False
    x = torch.zeros(
        (total_nodes, tree_list[0].x.shape[1]),
        device=device,
        dtype=tree_list[0].x.dtype,
    )
    if hasedgeattr:
        edge_attr = torch.zeros(
            (total_edges, tree_list[0].edge_attr.shape[1]),
            device=device,
            dtype=tree_list[0].edge_attr.dtype,
        )
    edge_index = torch.zeros((2, total_edges), device=device, dtype=torch.long)
    node_start = 0
    edge_start = 0
    for tree in tree_list:
        num_nodes = tree.x.shape[0]
        idxs = torch.arange(num_nodes) + node_start
        x[idxs, :] = tree.x
        num_edges = tree.edge_index.shape[1]
        idxs = torch.arange(num_edges) + edge_start
        if hasedgeattr:
            edge_attr[idxs, :] = tree.edge_attr
        edge_index_extract = tree.edge_index
        edge_index_extract = edge_index_extract + node_start
        edge_index[:, idxs] = edge_index_extract
        node_start += num_nodes
        edge_start += num_edges
    if hasedgeattr:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=class_)
    else:
        data = Data(x=x, edge_index=edge_index, y=class_)
    return data


def roots_to_embed(data):
    nn = data.x.shape[0]
    root = torch.zeros((nn,), device=data.x.device)
    root_indices = torch.tensor(
        list(
            set(data.edge_index[1].tolist()).difference(
                set(data.edge_index[0].tolist())
            )
        )
    ).to(data.x.device)
    root[root_indices] = 1
    data.roots_to_embed = root
    return data


class myDataset(Dataset):
    def __init__(this, dataset_list, dataset_len=0, **kwargs):
        super().__init__()
        this.dataset_list = [list(d.values())[0] for d in dataset_list]
        this.dataset_len = max(len(this.dataset_list), dataset_len)
        this.rng = np.random.RandomState(seed=kwargs.get("seed", 0))
        freq, data = list(zip(*((d["freq"], d["data"]) for d in this.dataset_list)))
        probs = np.asarray(freq) / np.sum(freq)
        this.data = [
            data[this.rng.choice(np.arange(len(data)), p=probs)]
            for _ in range(this.dataset_len)
        ]
        this.dataset_len = len(this.dataset_list)
        this.data = data

    def len(this):
        return this.dataset_len

    def get(this, idx):
        return this.data[idx]


def get_dataloader(dataset_classwise, **kwargs):
    if isinstance(dataset_classwise, dict):
        dataset = (
            dataset_classwise[0] + dataset_classwise[1]
        )  # for now assumes only two classes named 0 and 1 exist
        mydataset = myDataset(dataset, **kwargs)
        if "dataset_len" in kwargs:
            del kwargs["dataset_len"]
        dataloader = DataLoader(mydataset, **kwargs)
    else:
        dataloader = DataLoader(dataset_classwise, **kwargs)
    return dataloader


# ---------------  PROCESS DATASET (LABELS)  ------------------
def get_label_maps(dataset):
    node_labels = set()
    node_labels_full = set()
    for d in dataset:
        node_label = d.y
        node_label_full = d.x
        [node_labels.add(n.item()) for n in node_label]
        [
            node_labels_full.add(tuple(node_label_full[n, :].tolist()))
            for n in range(len(node_label_full))
        ]
    node_label_map = {n: idx for idx, n in enumerate(node_labels)}
    node_label_map_full = {n: idx for idx, n in enumerate(node_labels_full)}
    return node_label_map, node_label_map_full


def process_labels(dataset, node_label_map, node_label_map_full):
    dataset_ = []
    for d in dataset:
        data = Data(
            x=torch.tensor(
                [node_label_map[n.item()] for n in d.y]
            ),  # since this goes into C++ and out comes just a string, the node features are lost
            original_x=d.x,
            edge_index=d.edge_index,
            y=d.y,
            node_attr=torch.tensor(
                [
                    node_label_map_full[tuple(d.x[n, :].tolist())]
                    for n in range(d.x.shape[0])
                ]
            ),
        )
        dataset_.append(data)
    return dataset_


def preprocess_dataset(dataset, full_dataset):
    node_label_map, node_label_map_full = get_label_maps(full_dataset)
    label_processed_dataset = process_labels(
        dataset,
        node_label_map,
        node_label_map_full,
    )
    return label_processed_dataset, node_label_map, node_label_map_full


def preprocess_dataset_test(
    dataset, node_label_map, edge_label_map, node_label_map_full
):
    label_processed_dataset = process_labels(
        dataset, node_label_map, edge_label_map, node_label_map_full
    )
    return label_processed_dataset


# ----------------  PROCESS CANONICAL LABELS  -----------------
def prettify_canonical_label(label):
    assert isinstance(label, str)
    tokens = label.split()
    label2 = f"<{tokens[0]},{tokens[1]}>"
    ptr = 2
    while ptr < len(tokens):
        if tokens[ptr] == "$":
            label2 += tokens[ptr]
            ptr += 1
        else:
            node_label = tokens[ptr]
            node_original_label = tokens[ptr + 1]
            edge_label = tokens[ptr + 2]
            ptr += 3
            label2 += f"<{node_label},{node_original_label},{edge_label}>"
    return label2


def canonical_label_to_naturals(dataset_labels):
    mapping = {}
    ctr = 0
    for graph_labels in dataset_labels:
        for label in graph_labels:
            if label not in mapping:
                mapping[label] = ctr
                ctr += 1
    return mapping


# ---------------  COMPUTE FREQUENT PATTERNS  -----------------
def tree_class_ctr(classes, dataset):
    tree_class_count = {}
    for class_, graph in zip(classes, dataset):
        for tree in graph:
            if tree not in tree_class_count:
                tree_class_count[tree] = {}
            if class_ not in tree_class_count[tree]:
                tree_class_count[tree][class_] = 0
            tree_class_count[tree][class_] += 1
    return tree_class_count


def get_invalid_trees(tree_class_count):
    trees = []
    for tree, ctr in tree_class_count.items():
        if len(ctr) != 1:
            trees.append(tree)
    return trees


def pyfpgrowth_wrapper(classwise, freq_thresholds):
    patterns = {}
    for class_, dataset in classwise.items():
        thresh = freq_thresholds[class_]
        class_patterns = pyfpgrowth.find_frequent_patterns(dataset, thresh)
        patterns[class_] = class_patterns
    return patterns


# -------------  </COMPUTE FREQUENT PATTERNS>  ----------------


def parse_canonical_label_bak(label):
    q = 0
    parsed = []
    idx = 0
    tmp = ""
    etmp = ""
    stack = []
    node_label_map = {}
    edge_label_map = {}
    nid = 0
    for idx in range(len(label) - 1):
        if q == 0:
            if label[idx].isnumeric():
                tmp += label[idx]
            elif label[idx] == "<":
                # create root here
                stack.append(nid)
                node_label_map[nid] = tmp
                nid += 1
                parsed.append({"node_label": tmp})
                q = 1
                tmp = ""
            elif label[idx] == "$":
                # create root here
                stack.append(nid)
                node_label_map[nid] = tmp
                break
        if q == 1:
            if label[idx] == "$":
                # this will be used to pop parent for creating actual edges later
                stack.pop()
                pass
            if label[idx] == "<":
                q = 2
        if q == 2:
            if label[idx].isnumeric():
                tmp += label[idx]
            elif label[idx] == ",":
                q = 3
        if q == 3:
            if label[idx].isnumeric():
                etmp += label[idx]
            if label[idx] == ">":
                parsed.append({"node_label": tmp, "edge_label": etmp})
                # create node here
                par = stack[-1]
                stack.append(nid)
                node_label_map[nid] = tmp
                # create edges here
                # (to, from)
                edge_label_map[(nid, par)] = etmp
                nid += 1
                tmp = ""
                etmp = ""
                q = 1
    assert q == 1, f"Parsing failed for input {label}"
    return node_label_map, edge_label_map


def parse_canonical_label(label):
    q = 0
    idx = 0
    tmp = ""
    otmp = ""
    etmp = ""
    stack = []
    node_label_map = {}
    node_label_map_original = {}
    edge_label_map = {}
    nid = 0
    for idx in range(len(label) - 1):
        if q == 0:
            if label[idx] == "<":
                q = 1
        elif q == 1:
            if label[idx].isnumeric():
                tmp += label[idx]
            elif label[idx] == ",":
                q = 2
        elif q == 2:
            if label[idx].isnumeric():
                otmp += label[idx]
            elif label[idx] == ">":
                # create root here
                stack.append(nid)
                node_label_map[nid] = tmp
                node_label_map_original[nid] = otmp
                tmp = ""
                otmp = ""
                nid += 1
                q = 3
        elif q == 3:
            if label[idx] == "<":
                q = 4
            elif label[idx] == "$":
                # pop stack here
                stack.pop()
        elif q == 4:
            if label[idx].isnumeric():
                etmp += label[idx]
            elif label[idx] == ",":
                q = 5
        elif q == 5:
            if label[idx].isnumeric():
                tmp += label[idx]
            elif label[idx] == ",":
                q = 6
        elif q == 6:
            if label[idx].isnumeric():
                otmp += label[idx]
            elif label[idx] == ">":
                # create new node and edge here
                par = stack[-1]
                stack.append(nid)
                node_label_map[nid] = tmp
                node_label_map_original[nid] = otmp
                edge_label_map[(nid, par)] = etmp
                # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/convert.html#from_networkx
                # in the source, edge_index[0,:] is sources and edge_index[1,:] is destinations
                nid += 1
                tmp = ""
                otmp = ""
                etmp = ""
                q = 3
    assert q == 3, f"Parsing failed for input {label}"
    return node_label_map, edge_label_map, node_label_map_original


def get_data(
    node_label_map,
    edge_label_map,
    node_label_map_original,
    node_label_map_orig,
    edge_label_map_orig,
    node_label_map_full,
    class_,
):
    i2n = {v: k for k, v in node_label_map_orig.items()}
    i2n_o = {v: k for k, v in node_label_map_full.items()}
    i2e = {v: k for k, v in edge_label_map_orig.items()}
    # features = [list(i2n_o[int(n)]) for idx, n in node_label_map_original.items()]
    # node_labels = [i2n[int(n)] for idx, n in node_label_map.items()]
    features, node_labels = [], []
    for idx, n in node_label_map_original.items():
        feature = list(i2n_o[int(n)])
        features.append(feature)
        #
        node_label = i2n[int(node_label_map[idx])]
        node_labels.append(node_label)
    #
    edge_index_row = []
    edge_index_col = []
    edge_attr = []
    for k, v in edge_label_map.items():
        edge_index_row.append(k[0])  # k[0] is child (source)
        edge_index_col.append(k[1])  # k[1] is parent (destination)
        edge_attr.append(i2e[int(v)])
    edge_index = [edge_index_row, edge_index_col]
    data = Data(
        x=torch.tensor(features),
        edge_index=torch.tensor(edge_index),
        edge_attr=torch.tensor(edge_attr),
        node_labels=torch.tensor(node_labels),
    )
    return data


def get_previous_split(si: str):
    """
    Given si like '1+2+3', return '1+2'.
    If si has no plus, return None.
    """
    parts = si.split("+")
    if len(parts) <= 1:
        return None
    return "+".join(parts[:-1])
