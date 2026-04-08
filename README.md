# Popularity predictor (baseline)

Binary classification scaffold: **Billboard Hot 100** for chart outcomes and labels, **Kaggle** for Spotify-style **audio features**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment

- Copy `.env.example` → `.env`.
- **Kaggle**: [Account → API → Create New Token](https://www.kaggle.com/settings) → save `kaggle.json` to `~/.kaggle/` **or** set `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env`.
- Set `KAGGLE_DATASET` to a slug like `owner/dataset-name` (the dataset you choose should include track name, artist, and audio feature columns).

### Top Hits Spotify notebook (API)

The notebook [Top Hits Spotify from 2000–2019](https://www.kaggle.com/code/youssefabdelghfar/top-hits-spotify-from-2000-2019) is a **Kaggle Kernel**. The CSV comes from an **input dataset**, not from the notebook file itself. This repo uses the Kaggle API to:

1. **`kernels_pull`** that kernel with **`metadata=True`** → read `kernel-metadata.json` → discover attached **dataset slug(s)**.
2. **`dataset_download_files`** for each slug (or a known fallback dataset if the metadata shape is empty).

Optional env overrides: `KAGGLE_TOP_HITS_KERNEL`, `KAGGLE_TOP_HITS_DATASET_FALLBACK`.

## Build training data (ETL)

1. **Download** a Kaggle dataset 

2. **Merge** Billboard chart rows for a given week with rows from the Kaggle CSV by matching track + artist names.

```bash
# Preview (no network)
python -m src.etl.build_dataset --chart-date 2023-08-12 --dry-run

# Top Hits Spotify (2000–2019): pull kernel metadata + download dataset CSV via Kaggle API
python -m src.etl.build_dataset --chart-date 2023-08-12 --fetch-top-hits-kernel --out data/merged_dataset.csv

# Download a generic Kaggle dataset (needs Kaggle credentials + KAGGLE_DATASET in .env)
python -m src.etl.build_dataset --chart-date 2023-08-12 --download-kaggle --out data/merged_dataset.csv

# Or use a local CSV path
python -m src.etl.build_dataset --chart-date 2023-08-12 --kaggle-csv path/to/your_audio_features.csv --out data/merged_dataset.csv
```

**Hit label**: `is_hit = 1` when `rank <= 100` on the Hot 100 for that chart week (same as a Top 100 appearance). Rows are **positives by construction**; add a separate **non-hit** sample for a full binary training set.

### Full pipeline: Top Hits (Kaggle API) + Billboard per song

Downloads the [Top Hits Spotify notebook’s dataset](https://www.kaggle.com/code/youssefabdelghfar/top-hits-spotify-from-2000-2019) via the Kaggle API, keeps **audio features** and **Spotify `spotify_popularity`**, then matches each row to **Hot 100** snapshots from `billboard.py` (rank, weeks on chart, peak, etc.).

**Align Billboard with the dataset (2000–2019)** — use `--billboard-year-range` instead of hand-picking dates:

| `--billboard-sample` | ~# of Hot 100 requests | Use when |
|----------------------|-------------------------|----------|
| `yearly` | 20 | Quick smoke test |
| `monthly` (default) | 240 | Good coverage vs runtime |
| `weekly` | ~1044 | Maximum coverage (slow; ~minutes with `--billboard-sleep`) |

```bash
# Same era as Top Hits (2000–2019): one chart Saturday per month
python -m src.etl.merge_kaggle_billboard \
  --fetch-top-hits-kernel \
  --billboard-year-range 2000-2019 \
  --billboard-sample monthly \
  --out data/kaggle_billboard_songs.csv
```

Or pass explicit weeks: `--chart-dates 2005-06-18,2010-01-09,...`  
Optional env: `BILLBOARD_YEAR_RANGE=2000-2019`, `BILLBOARD_SAMPLE=monthly`.

Output: one row per Kaggle song, plus `billboard_*` columns (`billboard_matched`, `billboard_rank_best`, `billboard_peak_rank_primary`, …). Songs with no Hot 100 match on any chosen week have `billboard_matched=0`.

## Dashboard & baseline model

```bash
python dash_app.py
python -m src.model.train --data data/sample_songs.csv --out artifacts
```

## Layout

| Piece | Role |
|--------|------|
| `src/apis/billboard_api.py` | Hot 100 snapshot for a date |
| `src/apis/kaggle_dataset.py` | Load/normalize Kaggle CSV; optional download |
| `src/apis/kaggle_top_hits.py` | Top Hits notebook: kernel pull + dataset download via API |
| `src/etl/build_dataset.py` | Join Billboard + Kaggle (Billboard-first slice) |
| `src/etl/merge_kaggle_billboard.py` | Kaggle Top Hits API + Billboard per song (full table) |
| `src/etl/billboard_date_range.py` | Build 2000–2019 chart Saturdays (yearly / monthly / weekly) |
| `src/etl/matching.py` | Track/artist name matching score |
| `src/modeling.py` | Logistic regression baseline |
| `dash_app.py` | EDA + train baseline in the browser |
