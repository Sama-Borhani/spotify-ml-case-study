
# Spotify ML Case Study 

An end-to-end machine learning case study exploring track-level Spotify data. This project focuses on understanding audio features to model collaborative tracks, studying the impact of hit songs on albums, and building a basic music recommendation system using unsupervised learning.

##  Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Models](#models)
- [Results](#results)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Examples](#examples)
- [Limitations & Future Work](#limitations--future-work)
- [Contributors](#contributors)
- [License](#license)

---

##  Introduction

This project investigates Spotify’s audio feature dataset through data cleaning, visualization, and predictive modeling. The main goals include:
- Understanding if a hit song boosts popularity for other album tracks.
- Predicting artist collaborations using classification models.
- Building a song recommender based on audio similarity.

The case study demonstrates practical use of regression, classification, and unsupervised learning in music intelligence.

---

##  Project Structure


spotify-ml-case-study/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ environment.yml
├─ data/
│  ├─ raw/              
│  └─ processed/        
├─ notebooks/           
├─ reports/
│  ├─ figures/          # Visualizations like ROC curves, boxplots
│  └─ results.md        # Summary of model metrics
├─ references/
│  └─ dataset.md        # Dataset link & schema notes
└─ src/
   ├─ data_prep.py             # Data cleaning and feature engineering
   ├─ eda.py                   # Exploratory Data Analysis functions
   ├─ modeling/
   │  ├─ logistic_collab.py    # Logistic regression model
   │  └─ random_forest_collab.py  # Random forest model
   └─ recommend/
      └─ knn_similar_tracks.py # KNN-based recommendation engine
````

---

##  Installation

You can install the required dependencies using either:

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate spotify-ml
```

### Option 2: Pip

```bash
pip install -r requirements.txt
```

---

##  Usage

### Run Data Preparation

```bash
python src/data_prep.py
```

### Run EDA

```bash
python src/eda.py
```

### Train Models

```bash
python src/modeling/logistic_collab.py
python src/modeling/random_forest_collab.py
```

### Recommend Similar Songs

```bash
python src/recommend/knn_similar_tracks.py
```

---

##  Features

* Cleans and engineers features like song duration, collaboration flag, and key mappings.
* Visual diagnostics: boxplots, Q-Q plots, Box-Cox transformations.
* Statistical analysis of collaboration patterns.
* Classification models with hyperparameter tuning.
* Song similarity search using KNN.

---

##  Models

| Model               | Purpose                       | AUC Score | Notes                   |
| ------------------- | ----------------------------- | --------- | ----------------------- |
| Linear Regression   | Impact of hit songs on albums | \~0.01 R² | Weak relationship found |
| Logistic Regression | Predict collaborations        | \~0.63    | Regularized, polynomial |
| Random Forest       | Predict collaborations        | \~0.90    | Best performer          |
| K-Nearest Neighbors | Recommend similar tracks      | N/A       | Exploratory recommender |

---

##  Results

* **Random Forest** outperformed all classifiers with an AUC \~0.90.
* **Logistic Regression** had moderate success with \~0.63 AUC.
* **Linear Regression** did not support the hypothesis that hits improve sibling track popularity.
* KNN model demonstrated practical usage of unsupervised learning.

---

##  Dependencies

Key libraries used:

* `pandas`, `numpy`, `matplotlib`, `seaborn`
* `scikit-learn`, `statsmodels`, `scipy`
* `jupyterlab` or `notebook` for interactive analysis

For full list, see [`requirements.txt`](requirements.txt) or [`environment.yml`](environment.yml).

---

## Configuration

* All data preprocessing scripts and feature transformation logic are centralized in `src/data_prep.py`.
* Modeling parameters are defined inside each model script.
* Class balancing and scaling handled within modeling pipelines.

---

##  Dataset

* Spotify Tracks Dataset
* \~114,000 tracks with features like:

  * `danceability`, `energy`, `liveness`, `valence`, etc.
* [Reference and schema notes here](references/dataset.md)

**Note**: Raw dataset is excluded from the repo due to size constraints.

---

##  Examples

* ROC curve comparison between Logistic Regression and Random Forest
* Feature importance from Random Forest
* Distribution plots for popularity, tempo, and more
* Residual plots for linear and weighted regression

---

##  Limitations & Future Work

* **Limitations**:

  * Class imbalance required resampling
  * Model assumptions not always satisfied
  * Dataset excluded listener behavior and release timing

* **Future Directions**:

  * Include sentiment or lyric embeddings
  * Explore deep learning and sequential models
  * Use listener play counts or temporal trends

