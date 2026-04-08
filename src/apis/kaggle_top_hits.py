from __future__ import annotations

import json
import re
from pathlib import Path

# Notebook: https://www.kaggle.com/code/youssefabdelghfar/top-hits-spotify-from-2000-2019
TOP_HITS_SPOTIFY_KERNEL = "youssefabdelghfar/top-hits-spotify-from-2000-2019"

# Same CSV the notebook uses as an *input dataset* (typical attachment for this kernel).
# Used if kernel metadata does not list dataset slugs in a parseable form.
TOP_HITS_SPOTIFY_DATASET_FALLBACK = "paradisejoy/top-hits-spotify-from-20002019"


def _slug_from_dataset_item(item: object) -> str | None:
    if isinstance(item, str):
        s = item.strip()
        if re.match(r"^[\w-]+/[\w-]+$", s):
            return s
        m = re.search(r"kaggle\.com/datasets/([\w-]+/[\w-]+)", s)
        if m:
            return m.group(1)
    if isinstance(item, dict):
        for key in ("ref", "slug", "datasetRef", "url", "title"):
            v = item.get(key)
            got = _slug_from_dataset_item(v) if v is not None else None
            if got:
                return got
    return None


def _looks_like_dataset_slug(s: str) -> bool:
    if "/" not in s or s.count("/") != 1:
        return False
    owner, name = s.split("/", 1)
    if not owner or not name or owner.isdigit():
        return False
    return bool(re.match(r"^[\w-]+$", owner)) and bool(re.match(r"^[\w-]+$", name))


def _walk_json_for_dataset_slugs(obj: object, out: list[str]) -> None:
    """Fallback: collect `owner/dataset` strings anywhere in the metadata JSON."""
    if isinstance(obj, dict):
        for v in obj.values():
            _walk_json_for_dataset_slugs(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_json_for_dataset_slugs(v, out)
    elif isinstance(obj, str) and _looks_like_dataset_slug(obj.strip()):
        out.append(obj.strip())


def parse_dataset_slugs_from_kernel_metadata(metadata_path: Path) -> list[str]:
    """Read `kernel-metadata.json` from `kernels_pull(..., metadata=True)` and extract dataset slugs."""
    data = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    slugs: list[str] = []
    for key in ("dataset_sources", "dataset_data_sources"):
        val = data.get(key)
        if not val:
            continue
        if isinstance(val, list):
            for item in val:
                s = _slug_from_dataset_item(item)
                if s:
                    slugs.append(s)
        elif isinstance(val, str):
            s = _slug_from_dataset_item(val)
            if s:
                slugs.append(s)

    if not slugs:
        extra: list[str] = []
        _walk_json_for_dataset_slugs(data, extra)
        slugs = extra

    # de-dupe, keep order
    seen: set[str] = set()
    out: list[str] = []
    for s in slugs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def find_first_csv_under(root: Path) -> Path | None:
    root = Path(root)
    matches = sorted(root.rglob("*.csv"))
    return matches[0] if matches else None


def download_top_hits_spotify_via_kaggle_api(
    *,
    download_root: Path,
    kernel: str | None = None,
    dataset_fallback: str | None = None,
) -> Path:
    """
    Fetch the *Top Hits Spotify (2000–2019)* data used by the Kaggle notebook, via the Kaggle API:

    1. `kernels_pull(..., metadata=True)` for the notebook → read attached dataset slug(s).
    2. `dataset_download_files` for each slug (or the known fallback dataset).

    Returns path to a `.csv` file to load with `load_kaggle_csv`.

    Requires Kaggle credentials (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` + `KAGGLE_KEY`).
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError("Install the `kaggle` package: pip install kaggle") from e

    kernel = (kernel or TOP_HITS_SPOTIFY_KERNEL).strip()
    dataset_fallback = (dataset_fallback or TOP_HITS_SPOTIFY_DATASET_FALLBACK).strip()

    download_root = Path(download_root)
    kernel_dir = download_root / "top_hits_spotify_kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.kernels_pull(kernel, path=str(kernel_dir), metadata=True, quiet=True)

    meta_path = kernel_dir / "kernel-metadata.json"
    slugs: list[str] = []
    if meta_path.exists():
        slugs = parse_dataset_slugs_from_kernel_metadata(meta_path)

    if not slugs:
        slugs = [dataset_fallback]

    data_root = download_root / "top_hits_spotify_data"
    data_root.mkdir(parents=True, exist_ok=True)

    for slug in slugs:
        safe = slug.replace("/", "__")
        target = data_root / safe
        target.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(slug, path=str(target), unzip=True)

    csv_path = find_first_csv_under(data_root)
    if csv_path is None:
        raise FileNotFoundError(
            f"No .csv found under {data_root} after downloading: {slugs}. "
            "Check Kaggle dataset layout or set KAGGLE_FILENAME after inspecting the zip."
        )
    return csv_path
