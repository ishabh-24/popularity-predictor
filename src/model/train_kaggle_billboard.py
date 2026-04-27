from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths
from ..data import DatasetSpec, load_dataset
from ..modeling import RandomForestTrainConfig, TrainConfig, save_artifacts, train_evaluate_baseline, train_evaluate_random_forest
from ..nn_explainability import NNExplainabilityConfig, run_nn_explainability
from ..nn_modeling import NeuralNetTrainConfig, save_nn_artifacts, train_evaluate_neural_net


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
    p.add_argument(
        "--model",
        type=str,
        choices=["logreg", "rf", "nn"],
        default="logreg",
        help="logreg = logistic regression (baseline); rf = random forest; nn = PyTorch Lightning HitNet.",
    )
    p.add_argument("--n-estimators", type=int, default=200, help="RandomForest n_estimators (only --model rf).")
    p.add_argument("--max-depth", type=int, default=20, help="RandomForest max_depth; use 0 for None (only --model rf).")
    p.add_argument(
        "--rf-recent-years-window",
        type=int,
        default=None,
        nargs="?",
        const=4,
        metavar="N",
        help=(
            "Only --model rf: optional year filter release_year >= max_year - N. "
            "Default (omit): no year filter (full CSV, more hits for train/test). "
            "Pass with no value (--rf-recent-years-window) to use N=4 like logistic regression."
        ),
    )
    p.add_argument(
        "--run-explainability",
        action="store_true",
        help="Only --model nn: run post-training global + local explainability and save artifacts.",
    )
    p.add_argument(
        "--xai-local-rows",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated row offsets from the filtered training frame for local IG (only --model nn).",
    )
    p.add_argument(
        "--xai-max-samples",
        type=int,
        default=512,
        help="Max rows sampled for permutation and IG baseline reference.",
    )
    p.add_argument(
        "--xai-permutation-repeats",
        type=int,
        default=5,
        help="Permutation repeats per transformed feature.",
    )
    p.add_argument(
        "--xai-ig-steps",
        type=int,
        default=64,
        help="Integrated Gradients interpolation steps.",
    )
    p.add_argument(
        "--xai-top-k-local",
        type=int,
        default=15,
        help="Top-k grouped local attributions saved to nn_local_integrated_gradients.csv.",
    )
    return p


def _parse_local_row_offsets(raw: str, n_rows: int) -> list[int]:
    if not raw.strip():
        return [0] if n_rows > 0 else []
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    uniq = sorted(set(vals))
    for v in uniq:
        if v < 0 or v >= n_rows:
            raise ValueError(f"Invalid --xai-local-rows index {v}; valid range is 0..{max(0, n_rows - 1)}")
    return uniq


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

    if args.model == "logreg":
        result = train_evaluate_baseline(df, spec=spec, cfg=TrainConfig(use_smote=not args.no_smote))
        saved = save_artifacts(
            out_dir,
            pipeline=result["pipeline"],
            metrics=result["metrics"],
            config=result["config"],
            pipeline_filename="baseline_pipeline.joblib",
        )
    elif args.model == "rf":
        md = None if args.max_depth == 0 else args.max_depth
        result = train_evaluate_random_forest(
            df,
            spec=spec,
            cfg=RandomForestTrainConfig(
                use_smote=not args.no_smote,
                n_estimators=args.n_estimators,
                max_depth=md,
                recent_years_window=args.rf_recent_years_window,
            ),
        )
        saved = save_artifacts(
            out_dir,
            pipeline=result["pipeline"],
            metrics=result["metrics"],
            config=result["config"],
            pipeline_filename="random_forest_pipeline.joblib",
            metrics_filename="metrics_random_forest.json",
            config_filename="train_config_random_forest.json",
        )
    else:
        result = train_evaluate_neural_net(df, spec=spec, cfg=NeuralNetTrainConfig())
        saved = save_nn_artifacts(
            out_dir,
            bundle=result["pipeline"],
            metrics=result["metrics"],
            config=result["config"],
        )
        if args.run_explainability:
            local_rows = _parse_local_row_offsets(args.xai_local_rows, len(result["X_all"]))
            xai_cfg = NNExplainabilityConfig(
                max_samples=args.xai_max_samples,
                permutation_repeats=args.xai_permutation_repeats,
                ig_steps=args.xai_ig_steps,
                top_k_local=args.xai_top_k_local,
                random_state=int(result["config"].get("random_state", 42)),
            )
            xai_saved = run_nn_explainability(
                result["pipeline"],
                X_eval=result["X_test"],
                y_eval=np.asarray(result["y_test"], dtype=np.int64),
                X_local=result["X_all"].iloc[local_rows],
                out_dir=out_dir,
                cfg=xai_cfg,
            )
            saved.update({f"xai_{k}": v for k, v in xai_saved.items()})

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

