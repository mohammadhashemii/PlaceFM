# PlaceFM

<div align=center><img src="https://github.com/mohammadhashemii/PlaceFM/figs/pipeline.png" width="500" /></div>


## Prepare Environments

Install from requirements

`pip install -r requirements.txt`




## FSQ-19M POI Graph Dataset

Follow the steps below to construct the U.S. POI dataset:


### Download data

#### 1. Raw US POI Data from Foursquare

To start, download the raw US POI data from the Foursquare dataset which contains over 20 million of POIs. The dataset can be downloaded from [google drive](https://drive.google.com/file/d/15S2bJ4KoJQnbwpTeFaVv1kxvy89XrjTh/view?usp=drive_link).

Then, place the raw file in `data/fsq/raw/`.


#### 2. Category Embeddings from SD-CEM

Download pretrained semantic category embeddings from the [SD-CEM](https://www.ijcai.org/proceedings/2024/0231.pdf) repository: [repo](https://github.com/2837790380/SD-CEM/tree/main/embeddings).

Then, place the embedding files inside: `data/SD-CEM`.

#### 3. Foursquare Category Hierarchy

Download the official Foursquare POI category hierarchy CSV: [website](https://docs.foursquare.com/data-products/docs/categories). The filename is `personalization-apis-movement-sdk-categories.csv`.

Then, place the file in `data/fsq/raw/`.

#### 4. Census ZCTA Boundaries

Download the U.S. Census ZIP Code Tabulation Area (ZCTA) shapefiles: [Census database](https://www2.census.gov/geo/tiger/TIGER2024/ZCTA520/).

Then, place the shape file in: `data/fsq/Census/`.

---

At the end, the `data/` directory structure should be something like this:

```
data/
├── fsq/
│   ├── raw/
│   │   ├── <US_raw_POI_data_file> (e.g. fsq_us.parquet)
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


To generate state-level embeddings using PlaceFM:

```
python train.py --dataset fsq --method placefm --state <state abbr> --clustering_method kmeans --verbose
```

You can set default configuration by using `configs/placefm/<dataset_name>.json`. The generated embeddings will be saved in `checkpoints/placefm`.


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



If you find this repo helpful, we would appreciate it if you could cite our paper.

```
@article{hashemi2025placefm,
  title={PlaceFM: A Training-free Geospatial Foundation Model of Places using Large-Scale Point of Interest Data},
  author={Hashemi, Mohammad and Amiri, Hossein and Zufle, Andreas},
  journal={arXiv preprint arXiv:2507.02921},
  year={2025}
}
```