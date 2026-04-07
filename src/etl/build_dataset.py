from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from ..apis.billboard_api import BillboardClient, BillboardConfig
from ..apis.spotify_api import (
    SpotifyClient,
    SpotifyConfig,
    normalize_spotify_audio_features,
    normalize_spotify_artist,
    normalize_spotify_track,
)
from ..etl.matching import MatchKey, simple_match_score


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build model-ready dataset from Billboard + Spotify (framework).")
    p.add_argument("--chart-date", type=str, required=True, help="Billboard Hot 100 chart date: YYYY-MM-DD")
    p.add_argument("--out", type=str, default="data/merged_dataset.csv", help="Output CSV path")
    p.add_argument("--dry-run", action="store_true", help="Do not call any external services; print planned steps.")
    p.add_argument("--spotify-limit", type=int, default=5, help="Spotify search candidates per Billboard row.")
    p.add_argument("--min-match", type=float, default=0.75, help="Minimum match score to accept Spotify candidate.")
    return p


def _need_env(var: str) -> str:
    v = os.getenv(var, "").strip()
    if not v:
        raise RuntimeError(f"Missing required environment variable: {var}")
    return v


def main() -> int:
    load_dotenv()
    args = build_arg_parser().parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("DRY RUN. Planned steps:")
        print(f"- Fetch Billboard chart hot-100 for date={args.chart_date}")
        print("- For each Billboard entry: search Spotify for best matching track")
        print("- Fetch Spotify audio features for matched tracks (batch)")
        print("- Fetch Spotify artist metadata for matched artists (batch/unique)")
        print("- Join into one table and write CSV")
        print("\nTo enable real runs, set in `.env`:")
        print("- SPOTIFY_CLIENT_ID=...")
        print("- SPOTIFY_CLIENT_SECRET=...")
        return 0

    # --- Billboard ---
    chart_name = os.getenv("BILLBOARD_CHART_NAME", "hot-100")
    bb = BillboardClient(BillboardConfig(chart_name=chart_name))
    chart_rows = bb.get_chart(args.chart_date)
    bb_df = pd.DataFrame(chart_rows)

    # --- Spotify auth ---
    sp_cfg = SpotifyConfig(
        client_id=_need_env("SPOTIFY_CLIENT_ID"),
        client_secret=_need_env("SPOTIFY_CLIENT_SECRET"),
    )
    sp = SpotifyClient(sp_cfg)

    matched_tracks: list[dict[str, Any]] = []
    candidates_for_af: list[str] = []
    artist_ids: set[str] = set()

    for _, row in bb_df.iterrows():
        key_bb = MatchKey.from_row(row.get("track_name"), row.get("artist_name"))
        spotify_limit = int(os.getenv("SPOTIFY_SEARCH_LIMIT", str(args.spotify_limit)))
        candidates = sp.search_track(track_name=row["track_name"], artist_name=row["artist_name"], limit=spotify_limit)

        best = None
        best_score = -1.0
        for c in candidates:
            norm = normalize_spotify_track(c)
            key_sp = MatchKey.from_row(norm.get("track_name"), norm.get("artist_name"))
            score = simple_match_score(key_bb, key_sp)
            if score > best_score:
                best_score = score
                best = c

        base = row.to_dict()
        base["match_score"] = float(best_score)

        min_match = float(os.getenv("ETL_MIN_MATCH_SCORE", str(args.min_match)))
        if best is None or best_score < min_match:
            matched_tracks.append(base)
            continue

        norm_track = normalize_spotify_track(best)
        matched = {**base, **norm_track}
        matched_tracks.append(matched)

        if norm_track.get("spotify_track_id"):
            candidates_for_af.append(norm_track["spotify_track_id"])
        if norm_track.get("spotify_artist_id"):
            artist_ids.add(norm_track["spotify_artist_id"])

    match_df = pd.DataFrame(matched_tracks)

    # --- Audio features (batch) ---
    af_rows = sp.get_audio_features(candidates_for_af)
    af_df = pd.DataFrame([normalize_spotify_audio_features(r) for r in af_rows if r])

    # --- Artist metadata (unique) ---
    artist_meta = []
    for aid in sorted(artist_ids):
        artist_meta.append(normalize_spotify_artist(sp.get_artist(aid)))
    artist_df = pd.DataFrame(artist_meta)

    # --- Merge ---
    merged = match_df.merge(af_df, on="spotify_track_id", how="left").merge(artist_df, on="spotify_artist_id", how="left")

    # --- Target + canonical columns for the rest of the repo ---
    # Hit definition (your finalized choice): a "hit" means the track appears on the Billboard Hot 100 (Top 100).
    # Since inputs are Hot 100 rows, this dataset contains *positives* by construction.
    # You'll still need a separate set of "miss" tracks (is_hit=0) to train a binary classifier.
    merged["hit_definition"] = "billboard_hot_100_appearance"
    merged["is_hit"] = (pd.to_numeric(merged.get("rank"), errors="coerce") <= 100).astype("Int64")

    # Convenience: keep chart-derived columns in a stable shape (useful for later aggregation across weeks).
    merged["chart_rank"] = pd.to_numeric(merged.get("rank"), errors="coerce")
    merged["chart_weeks_on"] = pd.to_numeric(merged.get("weeks_on_chart"), errors="coerce")
    merged["chart_peak_rank"] = pd.to_numeric(merged.get("peak_rank"), errors="coerce")
    merged["chart_last_week_rank"] = pd.to_numeric(merged.get("last_week_rank"), errors="coerce")
    merged = merged.rename(
        columns={
            "spotify_popularity_track": "track_popularity",
        }
    )

    merged.to_csv(out_path, index=False)
    print(f"Wrote merged dataset: {out_path} (rows={len(merged)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

