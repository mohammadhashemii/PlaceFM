# PlaceFM

## POI Dataset Construction

Follow the steps below to construct the U.S. POI dataset:
---

# Prepare Environments

## Install from requirements
`pip install -r requirements.txt`

---

## Download data

### 1. Raw POI Data from [F-OSM](https://github.com/onspatial/f-osm)

To start, download the US raw POI data from the F-OSM dataset. The dataset can be downloaded from [google drive](https://drive.google.com/file/d/15S2bJ4KoJQnbwpTeFaVv1kxvy89XrjTh/view?usp=drive_link).

Then, place the raw file in `data/f-osm/raw/`.


### 2. Category Embeddings from SD-CEM

Download pretrained semantic category embeddings from the [SD-CEM](https://www.ijcai.org/proceedings/2024/0231.pdf) repository: [repo](https://github.com/2837790380/SD-CEM/tree/main/embeddings).

Then, place the embedding files inside: `data/SD-CEM`.

### 3. Foursquare Category Hierarchy

Download the official Foursquare POI category hierarchy CSV: [website](https://docs.foursquare.com/data-products/docs/categories). The filename is `personalization-apis-movement-sdk-categories.csv`.

Then, place the file in `data/f-osm/raw/`.

### 4. Census ZCTA Boundaries

Download the U.S. Census ZIP Code Tabulation Area (ZCTA) shapefiles: [Census database](https://www2.census.gov/geo/tiger/TIGER2024/ZCTA520/).

Then, place the shape file in: `data/f-osm/Census/`.

---

At the end, the `data/` directory structure should be something like this:

```
data/
├── f-osm/
│   ├── raw/
│   │   ├── <US_raw_POI_data_file> (e.g. us_pois_clean.csv)
│   │   └── personalization-apis-movement-sdk-categories.csv
│   └── Census/
│       └── <ZCTA_shapefile>
└── SD-CEM/
    └── <embedding_files> (e.g. SD-CEM#US#30.csv)
```


## Generate Embeddings via Foundation Model

First, change your directory to `placefm/`:

```
cd placefm
```


### Baselines 

#### 1. [HGI](https://www.sciencedirect.com/science/article/abs/pii/S0924271622003148): 

To train the HGI model and generate region embeddings:

```
python train.py --dataset f-osm --method hgi --city <city name> --verbose
```

The generated embeddings will be saved in `checkpoints/`.


## Evaluate on Geospatial Donwstream Tasks

To evaluate the effectiveness of generated region embeddings, we've implemented three downstream tasks:

### Downstream Tasks

1. **Population Density (PD) prediction**  
    Predict the population density of each US zipcode.

2. **Median House Price (HP) Prediction**  
    estimate the median house price for each US zipcode.

3. **Urban Functionality (UF)**  
    TODO

Run the following command to evaluate on abovementioned tasks. You can set the following parameters:

- `--embeddings`: Path to the region embeddings file.
- `--run_eval`: Number of evaluation runs (e.g., 10).
- `--dt_model`: Downstream task model, options: `rf` (Random Forest), `xgb` (XGBoost), etc.
- `--verbose`: Enable verbose output.

```
python test.py --embeddings <path to the embeddings> --run_eval 10 --dt_model rf --verbose
```

The results will be logged in `checkpoints/logs/`.