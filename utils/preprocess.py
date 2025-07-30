import pandas as pd
import numpy as np

def pad_zip(zip_val):
    try:
        zip_int = int(zip_val)
        zip_str = str(zip_int)
        while len(zip_str) < 5:
            zip_str = '0' + zip_str
        return zip_str
    except:
        return None  # or np.nan

def preprocess(csv_path):
    df = pd.read_csv(csv_path)
    

    # Drop unncessary columns
    df = df.drop(columns=["fsq_post_town", "fsq_country", "fsq_admin_region", "fsq_tel", "fsq_website",
                           "fsq_email", "fsq_facebook_id", "fsq_instagram", "fsq_po_box",
                           "fsq_twitter", "fsq_placemaker_url", "osm_type", "osm_admin_level"])

    # Keep only the row with the highest name_similarity_score per fsq_place_id
    df = df.sort_values('name_similarity_score', ascending=False).drop_duplicates(subset='fsq_place_id', keep='first')
    


    # Map full names or incorrect variants to valid state abbreviations
    df = df[df['fsq_region'] != 'Tamil Nadu']
    df = df[df['fsq_region'] != 'Baja Californa']

    fix_map = {
        'Calif': 'CA',
        'California': 'CA',
        'New York': 'NY',
        'Ny': 'NY',
        'Mi': 'MI',
        'Co': 'CO',
        'D.C.': 'DC'}
    
    df['fsq_region'] = df['fsq_region'].replace(fix_map)
    df['fsq_region'] = df['fsq_region'].str.upper()


    # extract osm_postcode from osm_address
    pattern = r'"postcode"\s*=>\s*"([^"]+)"'
    df['osm_postcode'] = df['osm_address'].str.extract(pattern)

    df = df.reset_index(drop=True)


    # # Extract and normalize ZIP code
    # zip_df["zip"] = zip_df["place"].str.extract(r"zip/(\d+)")
    # # zip_df["zip"] = zip_df["zip"].astype(str).str.zfill(5)

    # # Rename ZIP columns to use pdfm_ prefix (except zip key)
    # zip_df = zip_df.rename(columns={col: f"pdfm_{col}" for col in zip_df.columns if col != "zip"})

    # # === Merge POIs with ZIP-level embeddings ===
    # final_df = pd.merge(pois_df, zip_df, how="left", on="zip")

    # === Save Result ===
    
    df.to_csv("data/us_pois.csv", index=False)

if __name__ == "__main__":
    
    preprocess(csv_path="/scratch/mhashe4/repos/fm/data/fsq_osm_usa_name_filtered_5.csv")