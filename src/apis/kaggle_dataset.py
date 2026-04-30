from __future__ import annotations
import glob
import os
import polars as pl

#setting up aliases for the columns
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
POPULARITY_ALIASES = ("spotify_popularity", "popularity", "track_popularity", "song_popularity")
YEAR_ALIASES = ("year", "release_year", "yr")


def first_matching_column(df: pl.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def normalize_kaggle_audio_df(df: pl.DataFrame) -> pl.DataFrame:
    #This method renames the Kaggle CSV columns to the names expected by our model.

    out = df.clone()
    tc = first_matching_column(out, TRACK_ALIASES)
    ac = first_matching_column(out, ARTIST_ALIASES)
    rename: dict[str, str] = {}
    if tc and tc != "track_name":
        rename[tc] = "track_name"
    if ac and ac != "artist_name":
        rename[ac] = "artist_name"
    out = out.rename(rename)

    pop_col = first_matching_column(out, POPULARITY_ALIASES)
    if pop_col and pop_col != "spotify_popularity":
        out = out.rename({pop_col: "spotify_popularity"})

    yr_col = first_matching_column(out, YEAR_ALIASES)
    if yr_col and yr_col != "release_year":
        out = out.rename({yr_col: "release_year"})

    return out


def load_kaggle_csv(path: str | os.PathLike[str]) -> pl.DataFrame:
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Kaggle CSV not found: {path}")
    _, ext = os.path.splitext(path)
    if ext.lower() == ".csv":
        return pl.read_csv(path)
    raise ValueError(f"Expected a .csv file, got: {path}")


def download_kaggle_dataset(
    dataset_slug: str,
    *,
    download_dir: str | os.PathLike[str],
    unzip: bool = True,
) -> str:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError("Install the `kaggle` package: pip install kaggle") from e

    download_dir = os.fspath(download_dir)
    os.makedirs(download_dir, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=download_dir, unzip=unzip)
    return download_dir


def find_default_csv_in_dir(directory: str | os.PathLike[str]) -> str | None:
    #this picks the first .csv in the directory
    directory = os.fspath(directory)
    pattern = os.path.join(directory, "*.csv")
    csvs = sorted(glob.glob(pattern))
    return csvs[0] if csvs else None


def resolve_kaggle_csv_path(
    *,
    explicit_csv: str | os.PathLike[str] | None,
    download_root: str | os.PathLike[str],
    dataset_slug: str | None,
    filename_hint: str | None,
) -> str:
    dr = os.fspath(download_root)
    if explicit_csv:
        ep = os.fspath(explicit_csv)
        if os.path.isfile(ep):
            return ep

    if filename_hint:
        p = os.path.join(dr, filename_hint)
        if os.path.isfile(p):
            return p

    if dataset_slug:
        found = find_default_csv_in_dir(dr)
        if found:
            return found

    raise FileNotFoundError(
        "Could not find a Kaggle CSV."
    )
