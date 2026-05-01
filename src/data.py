"""
Data loading & preprocessing: The dataset in this repo merges the Kaggle "Top Hits" Spotify CSV 
(audio features + Spotify popularity) with Billboard Hot 100 info via name and artist matching. 
The normalization of raw Kaggle CSV columns (aliases, year/popularity column names, etc.) is done
by kaggle_dataset.py.

Outlier handling:
- Type and datetime parsing are handled
- Missing numeric audio features are tolerated
- Optional SMOTE oversampling to address class imbalance during training (`use_smote` flags)


Justification: Outliers are not explicitly handled in a separate step for our project. Instead, we use 
indirect methods like bounds on labels/probabilities to handle them e.g. median imputation with simple 
imputer which can handle skewed data well. Because of the nature of our problem (nonlinear correlations),
explicit outlier handling is not necessary since a few extreme feature values would not distort the fit
of models like RF the way that they would in logistic regression. While we have logistic regression as a
baseline, since it is not our main focus, we found that methods like imputation and standard scaler worked
well to standardize the data.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Iterable
import pandas as pd

@dataclass(frozen=True)
class DatasetSpec:

    target_col: str = "is_hit"
    id_cols: tuple[str, ...] = ("track_name", "artist_name")
    date_col: str = "release_date"

    numeric_cols: tuple[str, ...] = (
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "artist_popularity",
        "artist_followers",
    )

    categorical_cols: tuple[str, ...] = ("genre",)

    def expected_columns(self) -> set[str]:
        return set(self.id_cols) | {self.date_col, self.target_col} | set(self.numeric_cols) | set(
            self.categorical_cols
        )


def load_dataset(path: str | os.PathLike[str]) -> pd.DataFrame:
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    _, ext = os.path.splitext(path)
    ext_lower = ext.lower()
    if ext_lower == ".csv":
        return pd.read_csv(path)
    if ext_lower == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {ext} (use .csv or .parquet)")


def validate_dataset(df: pd.DataFrame, spec: DatasetSpec) -> list[str]:
    issues: list[str] = []
    missing = sorted(spec.expected_columns() - set(df.columns))
    if missing:
        issues.append(f"Missing expected columns {missing}")

    return issues


def coerce_types(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    out = df.copy()

    if spec.date_col in out.columns:
        out[spec.date_col] = pd.to_datetime(out[spec.date_col], errors="coerce")

    for col in spec.numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if spec.target_col in out.columns:
        out[spec.target_col] = pd.to_numeric(out[spec.target_col], errors="coerce").astype("Int64")

    return out


def add_time_features(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """
    Adds some contextual features from release_date 
    """
    out = df.copy()
    if spec.date_col not in out.columns:
        return out

    dt = pd.to_datetime(out[spec.date_col], errors="coerce")
    out["release_year"] = dt.dt.year
    out["release_month"] = dt.dt.month
    return out