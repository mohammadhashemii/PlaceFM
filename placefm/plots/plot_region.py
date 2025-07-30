import os
import sys
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import argparse
import torch
from shapely.geometry import LineString
import warnings
warnings.filterwarnings("ignore")

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

from placefm.dataset.f_osm import F_OSM


# === Args
parser = argparse.ArgumentParser()
parser.add_argument('--city', type=str, required=True, help='city name')
parser.add_argument('--postcode', type=str, required=False, help='ZIP Code to visualize')   
args = parser.parse_args()

# === Load POI data
data = F_OSM(root="../data", name="f-osm", args=args)

lat = data.lat_lon[:, 0].numpy()
lon = data.lat_lon[:, 1].numpy()

df = pd.DataFrame({
    "latitude": lat,
    "longitude": lon,
    "postcode": pd.Series(data.y),
    "category": pd.Series(data.category.values),
})

# === Filter POIs by ZIP code (if provided)
if hasattr(args, "postcode") and args.postcode is not None:
    target_postcode = int(args.postcode)
    df_zip = df[df["postcode"] == target_postcode]
    if df_zip.empty:
        print(f"No POIs found in ZIP code {target_postcode}")
        exit()

    # === Load ZIP code shapefile (ZCTA)
    zip_gdf = gpd.read_file('/scratch/mhashe4/repos/fm/data/tl_2024_us_zcta520/tl_2024_us_zcta520.shp').to_crs("EPSG:3395")
    zip_match = zip_gdf[zip_gdf["ZCTA5CE20"] == str(target_postcode).zfill(5)]
    if zip_match.empty:
        print(f"No ZIP boundary found for {target_postcode}")
        exit()

    # === Convert POIs to GeoDataFrame
    gdf_pois = gpd.GeoDataFrame(
        df_zip,
        geometry=gpd.points_from_xy(df_zip["longitude"], df_zip["latitude"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:3395")
else:
    # Plot all POIs
    gdf_pois = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:3395")
    zip_match = None
    df_zip = df  # for downstream code

# === Build Edge GeoDataFrame
poi_idx_in_zip = df_zip.index.tolist()
index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(poi_idx_in_zip)}

edge_geoms = []
edge_weights = []

for i in range(data.edge_index.size(1)):
    src = data.edge_index[0, i].item()
    dst = data.edge_index[1, i].item()

    if src in poi_idx_in_zip and dst in poi_idx_in_zip:
        src_geom = gdf_pois.loc[src, 'geometry']
        dst_geom = gdf_pois.loc[dst, 'geometry']
        line = LineString([src_geom, dst_geom])
        edge_geoms.append(line)
        edge_weights.append(data.edge_weight[i].item())

edge_gdf = gpd.GeoDataFrame(geometry=edge_geoms)
edge_gdf["weight"] = edge_weights
# === Plotting
fig, ax = plt.subplots(figsize=(36, 36))

if zip_match is not None:
    zip_match.boundary.plot(ax=ax, color='red', linewidth=2)
    edge_gdf.plot(ax=ax, linewidth=edge_gdf["weight"], color='gray', alpha=0.6)
    gdf_pois.plot(ax=ax, color='blue', markersize=20, alpha=1.0)
    ax.set_title(f"POIs & Edges in ZIP code {target_postcode}")
    plt.savefig(f"../figs/pois_edges_zip_{target_postcode}.png", dpi=300)
else:
    # Plot all ZIP boundaries in the city
    zip_gdf = gpd.read_file('/scratch/mhashe4/repos/fm/data/tl_2024_us_zcta520/tl_2024_us_zcta520.shp').to_crs("EPSG:3395")
    # Filter ZIPs that have POIs in the city
    zip_codes_in_city = df["postcode"].unique()
    zip_matches = zip_gdf[zip_gdf["ZCTA5CE20"].isin([str(z).zfill(5) for z in zip_codes_in_city])]
    zip_matches.boundary.plot(ax=ax, color='red', linewidth=1)
    edge_gdf.plot(ax=ax, linewidth=edge_gdf["weight"], color='gray', alpha=0.4)
    gdf_pois.plot(ax=ax, color='blue', markersize=10, alpha=1.0)

    # Plot the region connection edges using the region adjacency matrix
    import ipdb; ipdb.set_trace()
    sorted_zip_codes = sorted(zip_codes_in_city)
    for i in range(data.region_adjacency.size(0)):
        for j in range(data.region_adjacency.size(1)):
            if data.region_adjacency[i, j] == 1 and i != j:
                zip_i = sorted_zip_codes[i]
                zip_j = sorted_zip_codes[j]
                # Find the centroid of all POIs in each region (ZIP code)
                region_i_geom = gdf_pois[gdf_pois["postcode"] == zip_i].geometry
                region_j_geom = gdf_pois[gdf_pois["postcode"] == zip_j].geometry
                if not region_i_geom.empty and not region_j_geom.empty:
                    region_i_centroid = region_i_geom.unary_union.centroid
                    region_j_centroid = region_j_geom.unary_union.centroid
                    line = LineString([region_i_centroid, region_j_centroid])
                    ax.plot(*line.xy, color='green', linewidth=0.8, alpha=0.7)

    ax.set_title(f"POIs & Edges in {args.city}")
    plt.savefig(f"../figs/pois_edges_{args.city}.png", dpi=300)



plt.axis('off')
plt.tight_layout()
plt.show()
