from __future__ import annotations
import argparse
import os

import numpy as np
import pandas as pd
from joblib import load

from ..data import DatasetSpec, load_dataset
from ..modeling import TrainConfig, prepare_xy_for_training
from ..nn_explainability import NNExplainabilityConfig, run_nn_explainability
from ..nn_modeling import HitNetClassifierBundle, NeuralNetTrainConfig


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run explainability for a trained HitNet NN bundle.")
    p.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to hitnet_bundle.joblib (defaults to artifacts/hitnet_bundle.joblib).",
    )
    p.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to kaggle_billboard_songs.csv (defaults to data/kaggle_billboard_songs.csv).",
    )
    p.add_argument("--out", type=str, default="", help="Output directory for explainability artifacts.")
    p.add_argument(
        "--local-rows",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated row offsets from the filtered frame for local IG.",
    )
    p.add_argument("--max-samples", type=int, default=512, help="Max rows sampled for global/local reference.")
    p.add_argument("--permutation-repeats", type=int, default=5, help="Permutation repeats per feature.")
    p.add_argument("--ig-steps", type=int, default=64, help="Integrated Gradients interpolation steps.")
    p.add_argument("--top-k-local", type=int, default=15, help="Top-k grouped local attributions to save.")
    return p


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "billboard_matched" not in out.columns:
        raise ValueError("Expected column 'billboard_matched' in dataset.")
    out["is_hit"] = pd.to_numeric(out["billboard_matched"], errors="coerce").fillna(0).astype(int)
    out["is_hit"] = out["is_hit"].clip(0, 1)

    if "release_date" not in out.columns and "release_year" in out.columns:
        yr = pd.to_numeric(out["release_year"], errors="coerce")
        out["release_date"] = pd.to_datetime(yr.astype("Int64").astype(str) + "-01-01", errors="coerce")
    return out


def parse_offsets(raw: str, n_rows: int) -> list[int]:
    if not raw.strip():
        return [0] if n_rows > 0 else []
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    uniq = sorted(set(vals))
    for v in uniq:
        if v < 0 or v >= n_rows:
            raise ValueError(f"Invalid --local-rows index {v}; valid range is 0..{max(0, n_rows - 1)}")
    return uniq


def main() -> int:
    args = build_arg_parser().parse_args()

    model_path = args.model
    data_path = args.data if args.data else "data/kaggle_billboard_songs.csv"
    out_dir = args.out if args.out else "artifacts"

    if not os.path.isfile(model_path):
        raise SystemExit(f"Model not found: {model_path}")

    bundle = load(model_path)
    if not isinstance(bundle, HitNetClassifierBundle):
        raise SystemExit("Expected a HitNetClassifierBundle. Provide a NN model artifact (hitnet_bundle.joblib).")

    df = load_dataset(data_path)
    df = prepare_target(df)

    spec = DatasetSpec(
        target_col="is_hit",
        id_cols=("track_name", "artist_name"),
        date_col="release_date",
        numeric_cols=(
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "duration_ms",
        ),
        categorical_cols=("genre",),
    )

    nn_cfg = NeuralNetTrainConfig()
    prep_cfg = TrainConfig(
        test_size=nn_cfg.test_size,
        random_state=nn_cfg.random_state,
        use_smote=False,
        max_audio_missing_frac=nn_cfg.max_audio_missing_frac,
        recent_years_window=nn_cfg.recent_years_window,
    )

    df_prep, numeric_features, categorical_features, y, _meta = prepare_xy_for_training(df, spec=spec, cfg=prep_cfg)
    X = df_prep[numeric_features + categorical_features]

    local_offsets = parse_offsets(args.local_rows, len(X))
    xai_cfg = NNExplainabilityConfig(
        max_samples=args.max_samples,
        permutation_repeats=args.permutation_repeats,
        ig_steps=args.ig_steps,
        top_k_local=args.top_k_local,
        random_state=nn_cfg.random_state,
    )

    saved = run_nn_explainability(
        bundle,
        X_eval=X,
        y_eval=np.asarray(y, dtype=np.int64),
        X_local=X.iloc[local_offsets],
        out_dir=out_dir,
        cfg=xai_cfg,
    )

    print("Saved explainability artifacts:")
    for k, v in saved.items():
        print(f"- {k}: {v}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
