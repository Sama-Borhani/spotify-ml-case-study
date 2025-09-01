
# Spotify ML Case Study 🎧

This repository explores **track-level Spotify audio features** with three main goals:

1. **Data Preparation** → cleaning raw Spotify track data and engineering simple features.  
2. **Modeling** → predicting whether a track is a *collaboration* using Logistic Regression and Random Forest.  
3. **Recommendation** → building a simple **KNN “similar tracks” system** based on audio features.  

The project demonstrates an **end-to-end ML workflow**: raw data → preprocessing → exploratory analysis → modeling → reporting results.

---

##  Project Structure
```

spotify-ml-case-study/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ environment.yml
├─ data/
│  └─ raw/              
├─ notebooks/           
├─ reports/
│  ├─ figures/         
│  └─ results.md        
├─ references/
│  └─ dataset.md      
└─ src/
├─ data\_prep.py            
├─ eda.py                    
├─ modeling/
│  ├─ logistic\_collab.py    
│  └─ random\_forest\_collab.py
└─ recommend/
└─ knn\_similar\_tracks.py  

````

---

##  Setup

### Option A — virtualenv
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
````

### Option B — conda

```bash
conda env create -f environment.yml
conda activate spotify-ml
```

---

##  Usage

### 1. Prepare the data

```bash
python src/data_prep.py \
  --in data/raw/dataset.csv \
  --out data/processed/clean.csv
```

This script:

* Drops duplicates
* Converts `duration_ms` → `duration_mins`
* Flags collaborations (multiple artists separated by `;`)
* Filters out spoken-word outliers (`speechiness >= 0.66`)

---

### 2. Exploratory Data Analysis

```bash
python src/eda.py \
  --data data/processed/clean.csv
```

Generates histograms for key audio features and saves them in `reports/figures/`.

---

### 3. Run Models

```bash
# Logistic Regression baseline
python src/modeling/logistic_collab.py \
  --data data/processed/clean.csv \
  --out reports/results.md

# Random Forest classifier
python src/modeling/random_forest_collab.py \
  --data data/processed/clean.csv \
  --out reports/results.md
```

Each script appends results (AUC, Accuracy) to `reports/results.md`.

---

### 4. Similar Track Recommendations

```bash
python src/recommend/knn_similar_tracks.py \
  --data data/processed/clean.csv \
  --track "Bad Guy" \
  --k 10 \
  --out reports/results.md
```

Appends a list of the *k* most similar tracks by numeric audio features.

---

##  Results (Highlights)

* **Data Prep** → added features like `duration_mins` and `collaboration`.
* **Modeling** →

  * Logistic Regression: baseline, \~0.63 AUC
  * Random Forest: stronger model, \~0.90 AUC (danceability & acousticness important)
* **KNN Demo** → returns intuitive “similar tracks” lists.

---

##  Data

* Source: Spotify audio features dataset (track-level).
* Features include: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, etc.
* Engineered features: `duration_mins`, `collaboration`.
* See `references/dataset.md` for provenance & schema.
* This project uses the [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) hosted on Hugging Face 🤗.  
The dataset contains track-level audio features (danceability, energy, loudness, speechiness, acousticness, etc.) along with metadata such as track name, artist(s), and album.  

For this repository, a raw CSV file is placed in `data/` 
For full details, feature descriptions, and the complete dataset, please refer to the [Hugging Face dataset page](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset).
---

##  Contributing

This is a personal case study project. Issues and suggestions are welcome.

