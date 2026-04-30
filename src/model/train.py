from __future__ import annotations

import argparse

from ..data import DatasetSpec, load_dataset
from ..modeling import TrainConfig, save_artifacts, train_evaluate_baseline


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train baseline 'hit vs miss' model (no API calls).")
    p.add_argument("--data", type=str, default="", help="Path to dataset CSV/Parquet.")
    p.add_argument("--out", type=str, default="", help="Output artifacts directory.")
    p.add_argument("--no-smote", action="store_true", help="Disable SMOTE oversampling.")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    data_path = args.data if args.data else "data/sample_songs.csv"
    out_dir = args.out if args.out else "artifacts"

    df = load_dataset(data_path)
    result = train_evaluate_baseline(
        df,
        spec=DatasetSpec(),
        cfg=TrainConfig(use_smote=not args.no_smote),
    )

    saved = save_artifacts(
        out_dir,
        pipeline=result["pipeline"],
        metrics=result["metrics"],
        config=result["config"],
    )

    print("Saved artifacts:")
    for k, v in saved.items():
        print(f"- {k}: {v}")
    print("\nKey metrics:")
    for k in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
        print(f"- {k}: {result['metrics'].get(k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

