"""
Build a song-level dataset:
  - Kaggle Top Hits Spotify (2000–2019): audio features + Spotify popularity (via API download).
  - Billboard Hot 100: chart rankings where name-matching finds songs on chart snapshots.

Use `--billboard-year-range 2000-2019` with `--billboard-sample yearly|monthly|weekly`
to align Billboard pulls with the dataset era (instead of hand-picking --chart-dates).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from ..apis.billboard_api import BillboardClient, BillboardConfig
from ..apis.kaggle_dataset import load_kaggle_csv, normalize_kaggle_audio_df, resolve_kaggle_csv_path
from ..apis.kaggle_dataset import download_kaggle_dataset as download_generic_dataset
from ..apis.kaggle_top_hits import (
    TOP_HITS_SPOTIFY_DATASET_FALLBACK,
    TOP_HITS_SPOTIFY_KERNEL,
    download_top_hits_spotify_via_kaggle_api,
)
from ..etl.billboard_date_range import billboard_dates_for_dataset_years, parse_year_range
from ..etl.matching import MatchKey, simple_match_score


def _parse_chart_dates(s: str) -> list[str]:
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    return [p for p in parts if p]


def _pick_primary_match(
    matches: list[tuple[float, dict[str, Any], str]],
) -> tuple[dict[str, Any], str] | tuple[None, None]:
    if not matches:
        return None, None
    # Prefer highest match score, then best (lowest) chart rank
    def sort_key(t: tuple[float, dict[str, Any], str]) -> tuple[float, float]:
        s, bb, _ = t
        rank = float(bb.get("rank") or 999)
        return (s, -rank)

    matches_sorted = sorted(matches, key=sort_key, reverse=True)
    best_s, bb, dt = matches_sorted[0]
    return bb, dt


def _aggregate_billboard(matches: list[tuple[float, dict[str, Any], str]]) -> dict[str, Any]:
    if not matches:
        return {
            "billboard_matched": 0,
            "billboard_snapshots_matched": 0,
        }
    ranks: list[float] = []
    peaks: list[float] = []
    weeks: list[float] = []
    for _, bb, _ in matches:
        ranks.append(float(bb.get("rank") or 999))
        peaks.append(float(bb.get("peak_rank") or 999))
        weeks.append(float(bb.get("weeks_on_chart") or 0))
    primary, _primary_dt = _pick_primary_match(matches)
    assert primary is not None
    return {
        "billboard_matched": 1,
        "billboard_snapshots_matched": len(matches),
        "billboard_rank_best": min(ranks) if ranks else None,
        "billboard_rank_worst": max(ranks) if ranks else None,
        "billboard_peak_rank_best": min(peaks) if peaks else None,
        "billboard_weeks_on_chart_max": max(weeks) if weeks else None,
        "billboard_rank_primary": primary.get("rank"),
        "billboard_weeks_on_chart_primary": primary.get("weeks_on_chart"),
        "billboard_peak_rank_primary": primary.get("peak_rank"),
        "billboard_last_week_rank_primary": primary.get("last_week_rank"),
        "billboard_chart_date_primary": primary.get("chart_date"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Kaggle Top Hits (API) + Billboard Hot 100 merge → one CSV per Kaggle song."
    )
    p.add_argument(
        "--chart-dates",
        type=str,
        default="",
        help="Comma-separated Hot 100 week dates (YYYY-MM-DD). Omit if using --billboard-year-range.",
    )
    p.add_argument(
        "--billboard-year-range",
        type=str,
        default="",
        help="Inclusive years aligned with Top Hits 2000–2019, e.g. 2000-2019 (use with --billboard-sample).",
    )
    p.add_argument(
        "--billboard-sample",
        type=str,
        choices=["yearly", "monthly", "weekly"],
        default="monthly",
        help="How many chart weeks to pull per year: yearly (~20), monthly (~240), weekly (~1040 for 2000–2019).",
    )
    p.add_argument(
        "--billboard-sleep",
        type=float,
        default=0.3,
        help="Seconds to sleep between Billboard HTTP requests (be polite to their servers).",
    )
    p.add_argument("--out", type=str, default="data/kaggle_billboard_songs.csv", help="Output CSV path")
    p.add_argument("--dry-run", action="store_true", help="Print steps only; no API calls.")
    p.add_argument(
        "--fetch-top-hits-kernel",
        action="store_true",
        help="Download Top Hits Spotify dataset via Kaggle API (kernel metadata + dataset files).",
    )
    p.add_argument(
        "--kaggle-csv",
        type=str,
        default="",
        help="Use local CSV instead of downloading (skip --fetch-top-hits-kernel).",
    )
    p.add_argument(
        "--download-kaggle",
        action="store_true",
        help="Download generic KAGGLE_DATASET from .env (not the Top Hits kernel flow).",
    )
    p.add_argument("--min-match", type=float, default=0.75, help="Track/artist match threshold [0–1].")
    return p


def _resolve_chart_dates(args: argparse.Namespace) -> list[str]:
    if args.chart_dates and args.chart_dates.strip():
        return _parse_chart_dates(args.chart_dates)
    if args.billboard_year_range and args.billboard_year_range.strip():
        y0, y1 = parse_year_range(args.billboard_year_range)
        return billboard_dates_for_dataset_years(y0, y1, args.billboard_sample)
    raise SystemExit(
        "Provide either --chart-dates YYYY-MM-DD,... or "
        "--billboard-year-range 2000-2019 [--billboard-sample yearly|monthly|weekly]"
    )


def main() -> int:
    load_dotenv()
    args = build_arg_parser().parse_args()
    if not args.chart_dates.strip() and not args.billboard_year_range.strip():
        env_yr = os.getenv("BILLBOARD_YEAR_RANGE", "").strip()
        if env_yr:
            args.billboard_year_range = env_yr
            env_sm = os.getenv("BILLBOARD_SAMPLE", "").strip()
            if env_sm in ("yearly", "monthly", "weekly"):
                args.billboard_sample = env_sm
    chart_dates = _resolve_chart_dates(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    download_root = Path(os.getenv("KAGGLE_DOWNLOAD_DIR", "data/kaggle_raw"))
    dataset_slug = os.getenv("KAGGLE_DATASET", "").strip()
    filename_hint = os.getenv("KAGGLE_FILENAME", "").strip() or None
    min_match = float(os.getenv("ETL_MIN_MATCH_SCORE", str(args.min_match)))

    if args.dry_run:
        print("DRY RUN — would:")
        print("  1) Load Kaggle Top Hits CSV (API or --kaggle-csv)")
        n = len(chart_dates)
        preview = chart_dates if n <= 8 else chart_dates[:5] + ["..."] + chart_dates[-3:]
        print(f"  2) Fetch Billboard Hot 100 for {n} week(s), e.g. {preview}")
        print("  3) For each Kaggle song, match Billboard rows; aggregate ranks / peak / weeks")
        print(f"  4) Write: {out_path}")
        return 0

    explicit_csv = Path(args.kaggle_csv) if args.kaggle_csv else None

    if args.fetch_top_hits_kernel:
        kernel = os.getenv("KAGGLE_TOP_HITS_KERNEL", TOP_HITS_SPOTIFY_KERNEL).strip()
        fallback = os.getenv("KAGGLE_TOP_HITS_DATASET_FALLBACK", TOP_HITS_SPOTIFY_DATASET_FALLBACK).strip()
        csv_path = download_top_hits_spotify_via_kaggle_api(
            download_root=download_root,
            kernel=kernel,
            dataset_fallback=fallback,
        )
    elif explicit_csv and explicit_csv.exists():
        csv_path = explicit_csv
    elif args.download_kaggle:
        if not dataset_slug:
            raise RuntimeError("Set KAGGLE_DATASET in `.env` when using --download-kaggle.")
        download_generic_dataset(dataset_slug, download_dir=download_root, unzip=True)
        csv_path = resolve_kaggle_csv_path(
            explicit_csv=None,
            download_root=download_root,
            dataset_slug=dataset_slug,
            filename_hint=filename_hint,
        )
    else:
        raise RuntimeError(
            "Use --fetch-top-hits-kernel (recommended), or --kaggle-csv /path/to.csv, "
            "or --download-kaggle with KAGGLE_DATASET."
        )

    kaggle_raw = load_kaggle_csv(csv_path)
    kaggle_df = normalize_kaggle_audio_df(kaggle_raw)
    if "track_name" not in kaggle_df.columns or "artist_name" not in kaggle_df.columns:
        raise ValueError(
            "Kaggle CSV needs track + artist columns. "
            f"Columns: {list(kaggle_df.columns)}"
        )

    chart_name = os.getenv("BILLBOARD_CHART_NAME", "hot-100")
    bb = BillboardClient(BillboardConfig(chart_name=chart_name))

    chart_dfs: list[pd.DataFrame] = []
    for i, d in enumerate(chart_dates):
        rows = bb.get_chart(d)
        chart_dfs.append(pd.DataFrame(rows))
        if args.billboard_sleep > 0 and i + 1 < len(chart_dates):
            time.sleep(args.billboard_sleep)

    flat_bb: list[tuple[MatchKey, dict[str, Any], str]] = []
    for bb_df in chart_dfs:
        for _, row in bb_df.iterrows():
            d = row.to_dict()
            key_bb = MatchKey.from_row(d.get("track_name"), d.get("artist_name"))
            flat_bb.append((key_bb, d, str(d.get("chart_date", ""))))

    rows_out: list[dict[str, Any]] = []

    for i, kg in kaggle_df.iterrows():
        key_kg = MatchKey.from_row(kg.get("track_name"), kg.get("artist_name"))
        matches: list[tuple[float, dict[str, Any], str]] = []
        for key_bb, bb_dict, chart_date in flat_bb:
            s = simple_match_score(key_kg, key_bb)
            if s >= min_match:
                matches.append((s, bb_dict, chart_date))

        best_score = max((m[0] for m in matches), default=None)
        agg = _aggregate_billboard(matches)
        base = kg.to_dict()
        base["kaggle_source_csv"] = str(csv_path)
        base["billboard_match_score_best"] = float(best_score) if best_score is not None else None
        base.update(agg)
        rows_out.append(base)

    merged = pd.DataFrame(rows_out)
    merged.to_csv(out_path, index=False)
    n_matched = int((merged["billboard_matched"] == 1).sum()) if "billboard_matched" in merged.columns else 0
    print(f"Wrote {out_path} (rows={len(merged)}, billboard_matched={n_matched})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
