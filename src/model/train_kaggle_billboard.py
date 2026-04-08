from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import Paths
from ..data import DatasetSpec, load_dataset
from ..modeling import TrainConfig, save_artifacts, train_evaluate_baseline


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train binary hit classifier from kaggle_billboard_songs.csv (hit=Billboard match)."
    )
    p.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to kaggle_billboard_songs.csv (defaults to data/kaggle_billboard_songs.csv).",
    )
    p.add_argument("--out", type=str, default="", help="Output artifacts directory (defaults to artifacts/).")
    p.add_argument("--no-smote", action="store_true", help="Disable SMOTE oversampling.")
    p.add_argument(
        "--label-scheme",
        type=str,
        choices=["0/1", "1/2"],
        default="0/1",
        help="How to display labels in printed output. Model always trains on 0/1 internally.",
    )
    return p


def _prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define hit label from this dataset.

    We treat `billboard_matched==1` as a 'hit' (made Hot 100 in any sampled weeks),
    and `billboard_matched==0` as 'miss'.
    """
    out = df.copy()
    if "billboard_matched" not in out.columns:
        raise ValueError("Expected column 'billboard_matched' in dataset.")
    out["is_hit"] = pd.to_numeric(out["billboard_matched"], errors="coerce").fillna(0).astype(int)
    out["is_hit"] = out["is_hit"].clip(0, 1)

    # Provide a real date column expected by the shared preprocessing code.
    # This Kaggle CSV has `release_year` but typically not `release_date`.
    if "release_date" not in out.columns and "release_year" in out.columns:
        yr = pd.to_numeric(out["release_year"], errors="coerce")
        out["release_date"] = pd.to_datetime(yr.astype("Int64").astype(str) + "-01-01", errors="coerce")
    return out


def main() -> int:
    args = build_arg_parser().parse_args()
    paths = Paths.default()

    data_path = Path(args.data) if args.data else (paths.data_dir / "kaggle_billboard_songs.csv")
    out_dir = Path(args.out) if args.out else paths.artifacts_dir

    df = load_dataset(data_path)
    df = _prepare_target(df)

    # Feature contract for this dataset (uses what's present; doesn't require everything).
    spec = DatasetSpec(
        target_col="is_hit",
        id_cols=("track_name", "artist_name"),
        date_col="release_date",
        numeric_cols=(
            # Kaggle audio features
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            # Extra useful numeric fields from this CSV
            "spotify_popularity",
            "duration_ms",
        ),
        categorical_cols=("genre",),
    )

    result = train_evaluate_baseline(df, spec=spec, cfg=TrainConfig(use_smote=not args.no_smote))
    saved = save_artifacts(out_dir, pipeline=result["pipeline"], metrics=result["metrics"], config=result["config"])

    print("Saved artifacts:")
    for k, v in saved.items():
        print(f"- {k}: {v}")

    m = result["metrics"]
    print("\nLabel definition:")
    if args.label_scheme == "1/2":
        print("- 1 = hit (billboard_matched==1)")
        print("- 2 = miss (billboard_matched==0)")
        print("  (trained internally as hit=1, miss=0)")
    else:
        print("- 1 = hit (billboard_matched==1)")
        print("- 0 = miss (billboard_matched==0)")

    print("\nKey metrics:")
    for k in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
        print(f"- {k}: {m.get(k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

