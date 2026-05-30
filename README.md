# Popularity Predictor

Predict whether a song becomes a **Billboard Hot 100 hit** using Spotify-style **audio features** and metadata. The project builds a labeled dataset from Kaggle + Billboard, trains several classifiers, and explores results in an interactive **Dash** dashboard with **SHAP** explainability.

## What it does

1. **Data (ETL)** — Download Top Hits Spotify audio features (Kaggle API), match each track to **Billboard Hot 100** chart snapshots, and write a song-level CSV with a binary label:
   - **`is_hit` / `billboard_matched = 1`** — song appeared on the Hot 100 in at least one sampled chart week  
   - **`0`** — no match on those weeks  

2. **Modeling** — Binary classification on features such as danceability, energy, tempo, duration, release timing, and genre:
   - **Logistic regression** (baseline): scaled numeric features, one-hot genre, optional **SMOTE**
   - **Random forest**: tree-friendly preprocessing; by default uses **all release years** in the CSV (more positive examples than the logreg year window)
   - **Neural net (HitNet)**: PyTorch Lightning MLP with optional hyperparameter search and **Captum**-based explainability

3. **Dashboard** — `dash_app.py` loads `data/kaggle_billboard_songs.csv`, shows EDA plots, trains a selected model in-browser, prints metrics, and displays a **SHAP** summary bar chart.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set credentials (do not commit `.env`):

- **Kaggle** — [API token](https://www.kaggle.com/settings) as `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`
- Optional: `KAGGLE_DATASET`, `KAGGLE_DOWNLOAD_DIR`, `BILLBOARD_YEAR_RANGE`, `BILLBOARD_SAMPLE`, `ETL_MIN_MATCH_SCORE`

### Use the bundled dataset

If you already have `data/kaggle_billboard_songs.csv`, skip ETL and go straight to training or the dashboard.

```bash
python dash_app.py
# Open http://127.0.0.1:8050/
```

### Build the dataset from scratch

Merge Kaggle Top Hits (2000–2019) with Billboard chart weeks:

```bash
python -m src.etl.merge_kaggle_billboard \
  --fetch-top-hits-kernel \
  --billboard-year-range 2000-2019 \
  --billboard-sample monthly \
  --out data/kaggle_billboard_songs.csv
```

Smaller smoke test: `--billboard-sample yearly`. Heavier coverage: `--billboard-sample weekly` (slow; many API calls).

Alternative entry points:

- `python -m src.etl.build_dataset` — single chart date + Kaggle CSV  
- Local CSV: `--kaggle-csv path/to/features.csv`

## Train models

Default data path: `data/kaggle_billboard_songs.csv`  
Default artifacts directory: `artifacts/`

```bash
# Logistic regression (baseline)
python -m src.model.train_kaggle_billboard --model logreg

# Random forest (all years by default; more hits in train/test)
python -m src.model.train_kaggle_billboard --model rf

# Neural net
python -m src.model.train_kaggle_billboard --model nn

# Hyperparameter search (logreg / rf / nn)
python -m src.model.train_kaggle_billboard --model rf --tune --tune-n-iter 20

# NN + post-training explainability (permutation + Integrated Gradients)
python -m src.model.train_kaggle_billboard --model nn --run-explainability
```

**Random forest year filter** (optional; logreg always uses a recent-year window in code):

- Omit `--rf-recent-years-window` → full CSV  
- `--rf-recent-years-window` alone → same ~4-year window as logreg  
- `--rf-recent-years-window 8` → custom span  

**Saved artifacts** (under `--out`, default `artifacts/`):

| Model | Main file | Metrics / config |
|--------|-----------|------------------|
| Logistic regression | `baseline_pipeline.joblib` | `metrics.json`, `train_config.json` |
| Random forest | `random_forest_pipeline.joblib` | `metrics_random_forest.json`, `train_config_random_forest.json` |
| Neural net | `hitnet_bundle.joblib` (+ checkpoints) | `metrics_nn.json`, `train_config_nn.json` |

### Predict on a song

```bash
python -m src.model.predict_kaggle_billboard --model-type logreg --track "Shape of You" --artist "Ed Sheeran"
python -m src.model.predict_kaggle_billboard --model-type rf --row 42
python -m src.model.predict_kaggle_billboard --model-type nn --explain-local --row 0
```

Use `--artifacts-dir` if models are not in `artifacts/`. Use `--model path/to/pipeline.joblib` to override paths.

### Explainability (CLI)

```bash
python -m src.model.explain_kaggle_billboard --model artifacts/hitnet_bundle.joblib
```

## Project layout

```
popularity-predictor/
├── dash_app.py                 # Dash UI: EDA, train, SHAP
├── data/
│   ├── kaggle_billboard_songs.csv   # Main training table (after ETL)
│   └── kaggle_raw/                  # Downloaded Kaggle sources (optional)
├── artifacts/                  # Trained models & metrics (gitignored)
├── src/
│   ├── data.py                 # DatasetSpec, loading, validation
│   ├── modeling.py             # Logistic regression & random forest pipelines
│   ├── nn_modeling.py          # HitNet (Lightning)
│   ├── nn_explainability.py    # Captum IG / permutation importance
│   ├── logreg_shap.py          # SHAP for sklearn pipelines
│   ├── random_forest_shap.py
│   ├── nn_shap.py
│   ├── apis/                   # Kaggle & Billboard clients
│   ├── etl/                    # merge_kaggle_billboard, build_dataset, matching
│   └── model/
│       ├── train_kaggle_billboard.py
│       ├── predict_kaggle_billboard.py
│       ├── explain_kaggle_billboard.py
│       └── train.py            # Small-sample baseline trainer
└── requirements.txt
```

## Preprocessing notes

Training applies shared steps (see `prepare_xy_for_training` in `src/modeling.py`):

- Coerce types, derive `release_month` / `release_dow` from `release_date`
- Drop rows with too much missing audio data (default: >50% of core Spotify audio columns)
- **Logistic regression**: keep songs with `release_year >= max(release_year) - 4` (recent window)
- **Random forest**: no year filter unless you set `--rf-recent-years-window`
- **SMOTE** on the training split when the minority class is large enough; disabled automatically if too few positives

Metrics such as **F1 / precision / recall** are computed for the **hit** class (`1`). With a small test set and heavy imbalance, those numbers can be unstable even when **ROC-AUC** looks reasonable.

## Requirements

Python 3.10+ recommended. Key dependencies: `dash`, `plotly`, `pandas`, `scikit-learn`, `imbalanced-learn`, `torch`, `lightning`, `captum`, `shap`, `polars`, `kaggle`, `billboard.py`.

## License / data

Chart data comes from [Billboard](https://www.billboard.com/) via `billboard.py`. Audio features come from Kaggle datasets referenced in the ETL scripts. Respect Kaggle and Billboard terms of use when downloading or redistributing data.
