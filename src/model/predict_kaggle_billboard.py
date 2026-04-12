from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import load

from ..config import Paths


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict hit/miss for a song using the trained pipeline.")
    p.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to trained joblib pipeline. If omitted, uses --artifacts-dir and --model-type.",
    )
    p.add_argument(
        "--model-type",
        type=str,
        choices=["logreg", "rf"],
        default="logreg",
        help="Which saved pipeline to load when --model is omitted: baseline_pipeline.joblib or random_forest_pipeline.joblib.",
    )
    p.add_argument(
        "--artifacts-dir",
        type=str,
        default="",
        help="Directory containing saved pipelines (default: project artifacts/).",
    )
    p.add_argument(
        "--data",
        type=str,
        default="data/kaggle_billboard_songs.csv",
        help="CSV used for lookup (when predicting by track/artist).",
    )
    p.add_argument("--track", type=str, default="", help="Track name to look up in CSV.")
    p.add_argument("--artist", type=str, default="", help="Artist name to look up in CSV.")
    p.add_argument(
        "--row",
        type=int,
        default=-1,
        help="Row index in CSV to predict (0-based). Overrides --track/--artist.",
    )
    return p


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def main() -> int:
    args = build_arg_parser().parse_args()

    paths = Paths.default()
    art = Path(args.artifacts_dir) if args.artifacts_dir else paths.artifacts_dir
    if args.model:
        model_path = Path(args.model)
    else:
        name = "baseline_pipeline.joblib" if args.model_type == "logreg" else "random_forest_pipeline.joblib"
        model_path = art / name
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    pipe = load(model_path)

    df = pd.read_csv(args.data)

    if args.row >= 0:
        if args.row >= len(df):
            raise SystemExit(f"--row {args.row} out of range (rows={len(df)})")
        row = df.iloc[[args.row]].copy()
    else:
        if not args.track:
            raise SystemExit("Provide --row or --track (and optionally --artist).")

        t = _norm(args.track)
        a = _norm(args.artist)

        cand = df[df["track_name"].astype(str).map(_norm).eq(t)]
        if args.artist:
            cand = cand[cand["artist_name"].astype(str).map(_norm).eq(a)]

        if cand.empty:
            raise SystemExit("No matching row found in CSV for that track/artist.")

        # If multiple matches exist, take the first.
        row = cand.iloc[[0]].copy()

    # Training pipeline expects the same feature columns; we pass the whole row and let the ColumnTransformer select.
    # Ensure release_date exists (train script synthesizes it; do same here).
    if "release_date" not in row.columns and "release_year" in row.columns:
        yr = pd.to_numeric(row["release_year"], errors="coerce")
        row["release_date"] = pd.to_datetime(yr.astype("Int64").astype(str) + "-01-01", errors="coerce")
    if "release_date" in row.columns:
        dt = pd.to_datetime(row["release_date"], errors="coerce")
        row["release_month"] = dt.dt.month
        row["release_dow"] = dt.dt.dayofweek

    # Predict
    pred = int(pipe.predict(row)[0])
    proba = None
    if hasattr(pipe, "predict_proba"):
        proba = float(pipe.predict_proba(row)[0][1])

    track = row.get("track_name", pd.Series(["?"])).iloc[0]
    artist = row.get("artist_name", pd.Series(["?"])).iloc[0]

    print(f"Song: {track} — {artist}")
    if proba is not None:
        print(f"Predicted hit (1=yes, 0=no): {pred}  |  P(hit)= {proba:.3f}")
    else:
        print(f"Predicted hit (1=yes, 0=no): {pred}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

