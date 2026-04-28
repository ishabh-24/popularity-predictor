from __future__ import annotations

from pathlib import Path

import polars as pl


# Column name aliases → canonical names used by DatasetSpec / modeling
TRACK_ALIASES = ("track_name", "track", "song", "name", "title", "track name", "song_name")
ARTIST_ALIASES = ("artist_name", "artist", "artists", "artist name", "artist(s)")
AUDIO_FEATURE_COLS = (
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
)

# Spotify / dataset popularity (Top Hits CSV often has `popularity`)
POPULARITY_ALIASES = ("spotify_popularity", "popularity", "track_popularity", "song_popularity")
YEAR_ALIASES = ("year", "release_year", "yr")


def _first_matching_column(df: pl.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def normalize_kaggle_audio_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Rename common Kaggle CSV column variants to the names expected by `src/data.py` / the model.
    Adds canonical `spotify_popularity` and `release_year` when aliases exist.
    """
    out = df.clone()
    tc = _first_matching_column(out, TRACK_ALIASES)
    ac = _first_matching_column(out, ARTIST_ALIASES)
    rename: dict[str, str] = {}
    if tc and tc != "track_name":
        rename[tc] = "track_name"
    if ac and ac != "artist_name":
        rename[ac] = "artist_name"
    out = out.rename(rename)

    pop_col = _first_matching_column(out, POPULARITY_ALIASES)
    if pop_col and pop_col != "spotify_popularity":
        out = out.rename({pop_col: "spotify_popularity"})

    yr_col = _first_matching_column(out, YEAR_ALIASES)
    if yr_col and yr_col != "release_year":
        out = out.rename({yr_col: "release_year"})

    return out


def load_kaggle_csv(path: Path) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Kaggle CSV not found: {path}")
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    raise ValueError(f"Expected a .csv file, got: {path}")


def download_kaggle_dataset(
    dataset_slug: str,
    *,
    download_dir: Path,
    unzip: bool = True,
) -> Path:
    """
    Download a Kaggle dataset (owner/dataset-name) into `download_dir`.

    Auth: set `KAGGLE_USERNAME` and `KAGGLE_KEY` in the environment, or place
    `kaggle.json` in `~/.kaggle/` (see Kaggle account → API → Create New Token).
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError("Install the `kaggle` package: pip install kaggle") from e

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=str(download_dir), unzip=unzip)
    return download_dir


def find_default_csv_in_dir(directory: Path) -> Path | None:
    """Pick the first `.csv` in the directory (shallow)."""
    directory = Path(directory)
    csvs = sorted(directory.glob("*.csv"))
    return csvs[0] if csvs else None


def resolve_kaggle_csv_path(
    *,
    explicit_csv: Path | None,
    download_root: Path,
    dataset_slug: str | None,
    filename_hint: str | None,
) -> Path:
    """
    Resolve which CSV to load: explicit path, or env hint, or first csv after download dir.
    """
    if explicit_csv and explicit_csv.exists():
        return explicit_csv

    if filename_hint:
        p = download_root / filename_hint
        if p.exists():
            return p

    if dataset_slug:
        # After `dataset_download_files(..., unzip=True)`, files often live directly under download_root
        found = find_default_csv_in_dir(download_root)
        if found:
            return found

    raise FileNotFoundError(
        "Could not find a Kaggle CSV. Set --kaggle-csv, or KAGGLE_FILENAME in .env, "
        "or download the dataset first."
    )
