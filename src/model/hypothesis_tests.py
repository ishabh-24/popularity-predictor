from __future__ import annotations
import os
import numpy as np
import polars as pl
from scipy import stats

"""This file contains our hypothesis tests for the dataset. This is a simple test to check if 
the features are significantly different between the hit and miss songs. The purpose of these texts was
exploratory for our initial look at the data before we built our models. It was not for actually training
the models."""

DEFAULT_FEATURES: tuple[str, ...] = (
    "danceability",
    "energy",
    "valence",
    "tempo",
    "loudness",
)

DATA_PATH: str | None = None
OUT_PATH: str | None = None
TARGET_COL: str = "is_hit"
FEATURES: tuple[str, ...] = DEFAULT_FEATURES


def load_table(path: str | os.PathLike[str]) -> pl.DataFrame:
    path = os.fspath(path)
    _, ext = os.path.splitext(path)
    ext_lower = ext.lower()
    if ext_lower == ".csv":
        return pl.read_csv(path)
    if ext_lower == ".parquet":
        return pl.read_parquet(path)
    raise ValueError(f"Unsupported file type: {ext} (use .csv or .parquet)")


def label_helper(df: pl.DataFrame, preferred: str) -> np.ndarray:
    if preferred in df.columns:
        return df.select(pl.col(preferred).cast(pl.Float64, strict=False).clip(0, 1)).to_series().to_numpy()
    if "billboard_matched" in df.columns:
        return (
            df.select(pl.col("billboard_matched").cast(pl.Float64, strict=False).clip(0, 1))
            .to_series()
            .to_numpy()
        )
    raise ValueError(
        f"Could not find target column {preferred!r} or fallback 'billboard_matched' in dataset."
    )


def cohens_d(hit: np.ndarray, miss: np.ndarray) -> float:
    n1, n0 = len(hit), len(miss)
    if n1 < 2 or n0 < 2:
        return float("nan")
    s1 = np.var(hit, ddof=1)
    s0 = np.var(miss, ddof=1)
    pooled = ((n1 - 1) * s1 + (n0 - 1) * s0) / (n1 + n0 - 2)
    if pooled <= 0 or not np.isfinite(pooled):
        return float("nan")
    return float((np.mean(hit) - np.mean(miss)) / np.sqrt(pooled))


def run_hypothesis_tests(df: pl.DataFrame, *, target_col: str, features: list[str]) -> pl.DataFrame:
    y = label_helper(df, target_col)
    out_rows: list[dict[str, float | int | str]] = []

    for feature in features:
        if feature not in df.columns:
            out_rows.append(
                {
                    "feature": feature,
                    "status": "missing_column",
                }
            )
            continue

        valid = df.select(
            [
                pl.col(feature).cast(pl.Float64, strict=False).alias("x"),
                pl.Series(name="y", values=y).cast(pl.Float64, strict=False),
            ]
        ).drop_nulls()
        hit = valid.filter(pl.col("y") == 1).get_column("x").to_numpy()
        miss = valid.filter(pl.col("y") == 0).get_column("x").to_numpy()

        if len(hit) < 2 or len(miss) < 2:
            out_rows.append(
                {
                    "feature": feature,
                    "status": "insufficient_samples",
                    "n_hit": int(len(hit)),
                    "n_miss": int(len(miss)),
                }
            )
            continue

        t_stat, t_p = stats.ttest_ind(hit, miss, equal_var=False, nan_policy="omit")
        u_stat, u_p = stats.mannwhitneyu(hit, miss, alternative="two-sided")
        d = cohens_d(hit, miss)

        out_rows.append(
            {
                "feature": feature,
                "status": "ok",
                "n_hit": int(len(hit)),
                "n_miss": int(len(miss)),
                "mean_hit": float(np.mean(hit)),
                "mean_miss": float(np.mean(miss)),
                "mean_diff_hit_minus_miss": float(np.mean(hit) - np.mean(miss)),
                "welch_t_stat": float(t_stat),
                "welch_t_p_value": float(t_p),
                "mannwhitney_u_stat": float(u_stat),
                "mannwhitney_u_p_value": float(u_p),
                "cohens_d": d,
            }
        )

    return pl.DataFrame(out_rows)


def main() -> int:
    data_path = DATA_PATH if DATA_PATH else "data/kaggle_billboard_songs.csv"
    out_path = OUT_PATH if OUT_PATH else "artifacts/hypothesis_tests.csv"
    out_parent = os.path.dirname(out_path) 
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    features = [f for f in FEATURES if f]
    if not features:
        raise SystemExit("No features configured. Set FEATURES to at least one column.")

    df = load_table(data_path)
    results = run_hypothesis_tests(df, target_col=TARGET_COL, features=features)
    results.write_csv(out_path)

    print(f"Saved hypothesis test results: {out_path}")
    ok = int(results.filter(pl.col("status") == "ok").height) if "status" in results.columns else 0
    print(f"Features tested successfully: {ok}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())