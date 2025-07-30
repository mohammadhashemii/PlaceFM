import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from matplotlib.colors import ListedColormap
import argparse

state_name_to_abbr = {
    'Alabama': 'AL', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
    'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY',
    'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI',
    'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE',
    'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI',
    'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
    'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
}


# === Load the merged DataFrame ===
df = pd.read_csv("/scratch/mhashe4/repos/fm/data/f-osm/raw/us_pois_clean.csv")

us_states = gpd.read_file('/scratch/mhashe4/repos/fm/data/cb_2024_us_state_20m/cb_2024_us_state_20m.shp').to_crs("EPSG:3395")
us_cities = gpd.read_file('/scratch/mhashe4/repos/fm/data/cb_2024_us_place_500k/cb_2024_us_place_500k.shp').to_crs("EPSG:3395")


# states
us_states = us_states[~us_states["NAME"].isin(["Alaska", "Hawaii"])]
us_states = us_states.reset_index(drop=True)
us_states["label_point"] = us_states.representative_point()
us_states["abbr"] = us_states["NAME"].map(state_name_to_abbr)

# cities
us_cities = us_cities[~us_cities["STATE_NAME"].isin(["Alaska", "Hawaii"])]
us_cities = us_cities.reset_index(drop=True)
us_cities["label_point"] = us_cities.representative_point()




parser = argparse.ArgumentParser()
parser.add_argument('--states', nargs='*', default=["all"], help='GA CA or "all"')
parser.add_argument('--cities', nargs='*', default=["all"], help='Woodstock or "all"')
args = parser.parse_args()
selected_states = args.states
selected_cities = args.cities


if "all" not in selected_states:
    df = df[df["fsq_region"].isin(selected_states)]
    us_states = us_states[us_states["abbr"].isin(selected_states)]

if "all" not in selected_cities:
    df = df[df["fsq_locality"].isin(selected_cities)]
    us_cities = us_cities[us_cities["NAME"].isin(selected_cities) & us_cities["STATE_NAME"].map(state_name_to_abbr).isin(selected_states)]



# === Ensure numeric types for plotting ===
df["fsq_latitude"] = df["fsq_latitude"].combine_first(df["osm_latitude"])
df["fsq_longitude"] = df["fsq_longitude"].combine_first(df["osm_longitude"])

df = df[~df["fsq_region"].isin(["AK", "HI"])]


gdf_pois = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["fsq_longitude"], df["fsq_latitude"]),
    crs="EPSG:4326"  # geographic coordinates
)
gdf_pois = gdf_pois.to_crs("EPSG:3395")



state_codes, state_labels = pd.factorize(gdf_pois["fsq_region"])
gdf_pois["state_code"] = state_codes

colors = plt.get_cmap("tab20").colors * 3
cmap = ListedColormap(colors[:len(state_labels)])



# Perform spatial join: find which polygon (state) each point belongs to
# pois_with_state_geom = gpd.sjoin(
#     gdf_pois,
#     us_states[["NAME", "abbr", "geometry"]],
#     how="left",
#     predicate="within"
# )
# pois_with_state_geom["match"] = pois_with_state_geom["fsq_region"] == pois_with_state_geom["abbr"]

# gdf_pois_clean = pois_with_state_geom[pois_with_state_geom["match"]].copy()
# gdf_outliers = pois_with_state_geom[~pois_with_state_geom["match"]]

# === Plot ===
fig, ax = plt.subplots(figsize=(28, 22))


us_cities.boundary.plot(ax=ax, color="gray")
if "all" in selected_cities:
    us_states.boundary.plot(ax=ax, color="gray")


gdf_pois.plot(
    ax=ax,
    column="state_code",
    cmap=cmap,
    markersize=3,
    alpha=0.6,
    legend=False
)

# Plot abbreviations on the map
for idx, row in us_states.iterrows():
    x, y = row["label_point"].x, row["label_point"].y
    abbr = row["abbr"]
    ax.text(x, y, abbr, fontsize=20, ha='center', va='center', fontweight='bold', color='black')


# Add title and labels
plt.title("US POIs Colored by State (Region)", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)


plt.tight_layout()
if "all" in selected_states:
    filename = "figs/us_pois_all_states.png"
else:
    state_part = "_".join(sorted(selected_states))
    city_part = "_".join(sorted(selected_cities))
    filename = f"figs/us_pois_{state_part}_{city_part}.png"

plt.savefig(filename)

# gdf_pois_clean.to_csv("/scratch/mhashe4/repos/fm/data/f-osm/raw/us_pois_clean.csv")
