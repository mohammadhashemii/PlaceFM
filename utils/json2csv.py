import json
import pandas as pd
from pprint import pprint

# === Load JSON File ===
json_path = "/scratch/mhashe4/repos/fm/data/fsq-osm/joined_data_0_0.json"
csv_path = "/scratch/mhashe4/repos/fm/data/fsq-osm/joined_data_0_0.csv"

with open(json_path, "r") as f:
    pois = json.load(f)

# === Process Each Entry ===
rows = []

for entry in pois:
    fsq_data = entry.get("fsq", {})
    if not fsq_data or fsq_data.get("country") != "US":
        continue

    osm_candidates = entry.get("osm", [])
    if not osm_candidates or not isinstance(osm_candidates, list):
        continue

    # === Select the OSM match with the highest address_similarity_score ===
    best_osm = None
    best_score = -1  # similarity scores are >= 0

    for osm in osm_candidates:
        score = osm.get("address_similarity_score")
        if score is not None and not pd.isna(score) and score > best_score and score > 0.5:
            best_osm = osm
            best_score = score

    if best_osm is None:
        continue  # skip if no valid match with similarity score

    # === Flatten fsq ===
    fsq_flat = {
        k if k.startswith("fsq_") else f"fsq_{k}": v
        for k, v in fsq_data.items()
    }

    # === Extract and pad ZIP code ===
    postcode = fsq_data.get("postcode")
    if pd.isna(postcode):
        continue
    try:
        zip_str = str(int(float(postcode)))
        while len(zip_str) < 5:
            zip_str = "0" + zip_str
    except (ValueError, TypeError):
        continue

    fsq_flat["zip"] = zip_str

    # === Flatten OSM ===
    osm_flat = {
        k if k.startswith("osm_") else f"osm_{k}": v
        for k, v in best_osm.items()
    }

    # === Flatten OSM address ===
    # address = best_osm.get("address", {})
    # address_flat = {
    #     f"osm_{k}": v for k, v in address.items()
    # } if isinstance(address, dict) else {}
    

    
    # === Merge and collect row ===
    row = {**fsq_flat, **osm_flat}

    rows.append(row)

# === Create DataFrame ===
df = pd.DataFrame(rows)

print(f"{len(df)} total POIs in US saved at {csv_path}!")
# === Save to CSV ===
df.to_csv(csv_path, index=False)
