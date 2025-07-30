import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder




def extract(csv_path):
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

        # Remove brackets and quotes, then split by whitespace
        cleaned = re.sub(r"[\'\[\]]", "", str(x)).strip()
        return cleaned.split()

    def get_category_levels_multi(id_list):
        lvl1_list, lvl2_list, lvl3_list = [], [], []

        if not isinstance(id_list, list):
            return [[], [], []]

        for cid in id_list:
            label = category_id_to_label.get(cid)
            if label:
                # Clean and split label up to 3 levels
                parts = label.replace('"', '').replace("'", '').replace('[', '').replace(']', '').strip().split(" > ")

                # Keep only first 3 levels and pad with None if needed
                parts = parts[:3] + [None] * (3 - len(parts))

                lvl1_list.append(parts[0])
                lvl2_list.append(parts[1])
                lvl3_list.append(parts[2])

        return [lvl1_list, lvl2_list, lvl3_list]


    df["fsq_category_ids"] = df["fsq_category_ids"].apply(parse_category_ids)
    df[["fsq_category_label_lvl1", "fsq_category_label_lvl2", "fsq_category_label_lvl3"]] = (df["fsq_category_ids"].apply(get_category_levels_multi).apply(pd.Series))

    features["category_lvl1"] = df["fsq_category_label_lvl1"].apply(lambda x: x[-1] if isinstance(x, list) and x else None)
    features["category_lvl2"] = df["fsq_category_label_lvl2"].apply(lambda x: x[-1] if isinstance(x, list) and x else None)
    features["category_lvl3"] = df["fsq_category_label_lvl3"].apply(lambda x: x[-1] if isinstance(x, list) and x else None)

    features = features[features["category_lvl1"].notna()].reset_index(drop=True)

    level_encodings = []
    for level in ["category_lvl1", "category_lvl2", "category_lvl3"]:
        col = features[level]
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoded = encoder.fit_transform(col.values.reshape(-1, 1))
        encoded_df = pd.DataFrame(
            encoded.toarray(),
            columns=[f"{level}_{cat}" for cat in encoder.categories_[0]]
        )
        encoded_df.index = features.index
        level_encodings.append(encoded_df)


    # 8. Concatenate one-hot encoded category levels to features
    features = pd.concat([features] + level_encodings, axis=1)
    

    import ipdb; ipdb.set_trace()



if __name__ == "__main__":
    
    extract(csv_path="/scratch/mhashe4/repos/fm/data/f-osm/raw/us_pois_clean.csv")