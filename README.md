
# Spotify ML Case Study 🎧

This project explores Spotify audio features with three main goals:

1. **Data Preparation** – clean and feature-engineer a track-level dataset.  
2. **Modeling** – predict collaborations using Logistic Regression and Random Forest.  
3. **Recommendation** – find similar tracks using a simple KNN approach.

---

##  Project Structure
```

spotify-ml-case-study/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ environment.yml
├─ data/
│  └─ raw/              # raw dataset (not committed)
├─ notebooks/           # for exploration
├─ reports/
│  ├─ figures/          # saved plots
│  └─ results.md        # metrics + recommendations appended here
├─ references/
│  └─ dataset.md        # dataset info
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

### Option A — pip + venv
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

### 1. Data Prep

```bash
python src/data_prep.py \
  --in data/raw/dataset.csv \
  --out data/processed/clean.csv
```

### 2. EDA (optional histograms)

```bash
python src/eda.py \
  --data data/processed/clean.csv
```

### 3. Models

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

### 4. KNN Similar Tracks

```bash
python src/recommend/knn_similar_tracks.py \
  --data data/processed/clean.csv \
  --track "Bad Guy" \
  --k 10 \
  --out reports/results.md
```

---



