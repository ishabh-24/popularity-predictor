from __future__ import annotations

import argparse
import os
from typing import Any

import polars as pl
from dotenv import load_dotenv

from ..apis.billboard_api import BillboardClient, BillboardConfig
from ..apis.kaggle_dataset import (
    AUDIO_FEATURE_COLS,
    download_kaggle_dataset,
    load_kaggle_csv,
    normalize_kaggle_audio_df,
    resolve_kaggle_csv_path,
)
from ..apis.kaggle_top_hits import (
    TOP_HITS_SPOTIFY_DATASET_FALLBACK,
    TOP_HITS_SPOTIFY_KERNEL,
    download_top_hits_spotify_via_kaggle_api,
)
from ..etl.matching import MatchKey, simple_match_score


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build model-ready dataset: Billboard (chart labels) + Kaggle CSV (Spotify-style audio features)."
    )
    p.add_argument("--chart-date", type=str, required=True, help="Billboard Hot 100 chart date: YYYY-MM-DD")
    p.add_argument("--out", type=str, default="data/merged_dataset.csv", help="Output CSV path")
    p.add_argument("--dry-run", action="store_true", help="Do not call external services; print planned steps.")
    p.add_argument(
        "--kaggle-csv",
        type=str,
        default="",
        help="Path to a local Kaggle-exported CSV (audio features). Overrides download.",
    )
    p.add_argument(
        "--download-kaggle",
        action="store_true",
        help="Download KAGGLE_DATASET from Kaggle API into data/kaggle_raw/ (requires credentials).",
    )
    p.add_argument(
        "--fetch-top-hits-kernel",
        action="store_true",
        help=(
            "Use Kaggle API for the Top Hits Spotify (2000–2019) notebook flow: "
            "pull kernel metadata (youssefabdelghfar/...), then download attached dataset CSV. "
            "Overrides --download-kaggle / generic KAGGLE_DATASET for this run."
        ),
    )
    p.add_argument("--min-match", type=float, default=0.75, help="Minimum match score to join Billboard row to Kaggle row.")
    return p


def best_kaggle_match(
    bb_key: MatchKey,
    kaggle_df: pl.DataFrame,
    *,
    tk_col: str = "_tk",
    ar_col: str = "_ar",
) -> tuple[dict[str, Any] | None, float]:
    """Return (best_row_dict, score) for best matching Kaggle row."""
    best_row: dict[str, Any] | None = None
    best_s = -1.0
    for row in kaggle_df.iter_rows(named=True):
        kg_key = MatchKey(track_norm=row[tk_col], artist_norm=row[ar_col])
        s = simple_match_score(bb_key, kg_key)
        if s > best_s:
            best_s = s
            best_row = row
    return best_row, best_s


def main() -> int:
    load_dotenv()
    args = build_arg_parser().parse_args()

    out_path = args.out
    out_parent = os.path.dirname(out_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    download_root = os.getenv("KAGGLE_DOWNLOAD_DIR", "data/kaggle_raw")
    dataset_slug = os.getenv("KAGGLE_DATASET", "").strip()
    filename_hint = os.getenv("KAGGLE_FILENAME", "").strip() or None

    if args.dry_run:
        print(f"- Fetch Billboard chart for date={args.chart_date} (see BILLBOARD_CHART_NAME)")
        print("- Load Spotify-style audio features from a Kaggle dataset CSV (local path or download)")
        print("- Match each Billboard track/artist to the best row in the Kaggle table")
        print("- Merge chart metrics + audio features; write is_hit (Top 100) + chart fields")
        print("\nConfigure `.env`:")
        print("- KAGGLE_USERNAME=...  KAGGLE_KEY=...  (or ~/.kaggle/kaggle.json)")
        print("- KAGGLE_DATASET=owner/dataset-name   (optional if you use --kaggle-csv)")
        print("- KAGGLE_FILENAME=your_file.csv       (optional; else first .csv in download dir)")
        print("\nTop Hits Spotify notebook (API):")
        print(f"- --fetch-top-hits-kernel  → kernel {TOP_HITS_SPOTIFY_KERNEL}")
        print(f"  optional env: KAGGLE_TOP_HITS_KERNEL, KAGGLE_TOP_HITS_DATASET_FALLBACK (default {TOP_HITS_SPOTIFY_DATASET_FALLBACK})")
        return 0

    explicit_csv = args.kaggle_csv if args.kaggle_csv else None

    if args.fetch_top_hits_kernel:
        kernel = os.getenv("KAGGLE_TOP_HITS_KERNEL", TOP_HITS_SPOTIFY_KERNEL).strip()
        fallback = os.getenv("KAGGLE_TOP_HITS_DATASET_FALLBACK", TOP_HITS_SPOTIFY_DATASET_FALLBACK).strip()
        csv_path = download_top_hits_spotify_via_kaggle_api(
            download_root=download_root,
            kernel=kernel,
            dataset_fallback=fallback,
        )
    else:
        if args.download_kaggle:
            if not dataset_slug:
                raise RuntimeError("Set KAGGLE_DATASET in `.env` when using --download-kaggle.")
            download_kaggle_dataset(dataset_slug, download_dir=download_root, unzip=True)

        csv_path = resolve_kaggle_csv_path(
            explicit_csv=explicit_csv,
            download_root=download_root,
            dataset_slug=dataset_slug or None,
            filename_hint=filename_hint,
        )

    kaggle_raw = load_kaggle_csv(csv_path)
    kaggle_df = normalize_kaggle_audio_df(kaggle_raw)

    if "track_name" not in kaggle_df.columns or "artist_name" not in kaggle_df.columns:
        raise ValueError(
            "Kaggle CSV must have recognizable track + artist columns. "
            f"Got columns: {list(kaggle_df.columns)}. "
            "See TRACK_ALIASES / ARTIST_ALIASES in src/apis/kaggle_dataset.py."
        )

    kaggle_df = kaggle_df.with_columns(
        [
            pl.col("track_name")
            .cast(pl.Utf8, strict=False)
            .map_elements(lambda x: MatchKey.from_row(x, None).track_norm, return_dtype=pl.Utf8)
            .alias("_tk"),
            pl.col("artist_name")
            .cast(pl.Utf8, strict=False)
            .map_elements(lambda x: MatchKey.from_row(None, x).artist_norm, return_dtype=pl.Utf8)
            .alias("_ar"),
        ]
    )

    # --- Billboard ---
    chart_name = os.getenv("BILLBOARD_CHART_NAME", "hot-100")
    bb = BillboardClient(BillboardConfig(chart_name=chart_name))
    chart_rows = bb.get_chart(args.chart_date)
    bb_df = pl.DataFrame(chart_rows)

    min_match = float(os.getenv("ETL_MIN_MATCH_SCORE", str(args.min_match)))

    rows_out: list[dict[str, Any]] = []

    for _, bb_row in bb_df.iterrows():
        key_bb = MatchKey.from_row(bb_row.get("track_name"), bb_row.get("artist_name"))
        best_row, best_s = best_kaggle_match(key_bb, kaggle_df)

        base: dict[str, Any] = dict(bb_row)
        base["kaggle_match_score"] = float(best_s)
        base["kaggle_csv"] = str(csv_path)

        if best_row is None or best_s < min_match:
            rows_out.append(base)
            continue

        audio_payload = {c: best_row.get(c) for c in AUDIO_FEATURE_COLS if c in kaggle_df.columns}
        extra = {
            "track_name_kaggle": best_row.get("track_name"),
            "artist_name_kaggle": best_row.get("artist_name"),
            **audio_payload,
        }
        merged_row = {**base, **extra}
        rows_out.append(merged_row)

    merged = pl.DataFrame(rows_out).with_columns(
        [
            pl.lit("billboard_hot_100_appearance").alias("hit_definition"),
            pl.when(pl.col("rank").cast(pl.Float64, strict=False).is_not_null())
            .then((pl.col("rank").cast(pl.Float64, strict=False) <= 100).cast(pl.Int64))
            .otherwise(None)
            .alias("is_hit"),
            pl.col("rank").cast(pl.Float64, strict=False).alias("chart_rank"),
            pl.col("weeks_on_chart").cast(pl.Float64, strict=False).alias("chart_weeks_on"),
            pl.col("peak_rank").cast(pl.Float64, strict=False).alias("chart_peak_rank"),
            pl.col("last_week_rank").cast(pl.Float64, strict=False).alias("chart_last_week_rank"),
        ]
    )

    merged.write_csv(out_path)
    print(f"Wrote merged dataset: {out_path} (rows={len(merged)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
