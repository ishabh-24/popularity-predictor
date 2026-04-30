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
    Adds some contextual features from release_date - does not fully account for temporal data though.
    """
    out = df.copy()
    if spec.date_col not in out.columns:
        return out

    dt = pd.to_datetime(out[spec.date_col], errors="coerce")
    out["release_year"] = dt.dt.year
    out["release_month"] = dt.dt.month
    return out