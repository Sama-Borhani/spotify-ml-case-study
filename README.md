
# Spotify ML Case Study 🎧

_A clean, reproducible ML repo analyzing track-level Spotify audio features to:_
1. Test whether a hit track lifts an album’s popularity.  
2. Predict collaborations using classification models.  
3. Build a simple KNN-based “similar tracks” recommendation demo.

---

##  Project Structure
```

spotify-ml-case-study/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt
├─ environment.yml
├─ pyproject.toml
├─ pre-commit-config.yaml
├─ .github/workflows/ci.yml
├─ data/
│  ├─ raw/              # raw dataset (git-ignored)
│  └─ processed/        # cleaned & feature-engineered data
├─ notebooks/           # exploratory notebooks
├─ reports/
│  ├─ figures/          # saved plots
│  └─ results.md        # summary of metrics
├─ references/
│  └─ dataset.md        # dataset info + schema
├─ src/
│  └─ spotify\_ml/
│     ├─ **init**.py
│     ├─ config.py
│     ├─ paths.py
│     ├─ data\_prep.py
│     ├─ eda.py
│     ├─ modeling/
│     │  ├─ logistic\_collab.py
│     │  └─ random\_forest\_collab.py
│     └─ recommend/
│        └─ knn\_similar\_tracks.py
└─ tests/
├─ test\_imports.py
└─ test\_smoke.py

````

---

##  Quickstart

### Option A — Conda
```bash
conda env create -f environment.yml
conda activate spotify-ml
pre-commit install
````

### Option B — venv + pip

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pre-commit install
```

---

##  Usage

### 1. Data prep

```bash
python -m spotify_ml.data_prep --in data/raw/dataset.csv --out data/processed/clean.parquet
```

### 2. Modeling

```bash
# Logistic regression baseline
python -m spotify_ml.modeling.logistic_collab --data data/processed/clean.parquet --out reports/results.md

# Random forest classifier
python -m spotify_ml.modeling.random_forest_collab --data data/processed/clean.parquet --out reports/results.md
```

### 3. Recommendations

```bash
python -m spotify_ml.recommend.knn_similar_tracks \
  --data data/processed/clean.parquet \
  --track "Bad Guy" \
  --k 10 \
  --out reports/results.md
```

---

##  Results (highlights)

* **Album Lift (OLS):** weak/insignificant — little evidence that a hit boosts other album tracks.
* **Collaboration Prediction:**

  * Logistic Regression → AUC \~0.63
  * Random Forest → AUC \~0.90 (danceability & acousticness key features)
* **KNN Recommender:** returns intuitive “similar track” lists based on audio features.

---

##  Reproducibility & Quality

* **Formatting / Linting:** black, ruff, isort
* **Type checking:** mypy (optional)
* **Pre-commit hooks:** auto-format on commit
* **Continuous Integration:** GitHub Actions runs lint + tests on push/PR

---

##  Data

* Source: Spotify audio features dataset (track-level).
* Features include: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, etc.
* Large datasets should be placed in `data/` but kept **out of git**.

---

##  Contributing

* Fork → branch → PR.
* Run `pre-commit` locally before committing.
* Keep functions <100 lines, include docstrings & type hints.

---

##  License

MIT — see [LICENSE](LICENSE) for details.

```

---

