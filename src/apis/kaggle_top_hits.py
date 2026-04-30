from __future__ import annotations
import json
import os
import re

TOP_HITS_SPOTIFY_KERNEL = "youssefabdelghfar/top-hits-spotify-from-2000-2019"
TOP_HITS_SPOTIFY_DATASET_FALLBACK = "paradisejoy/top-hits-spotify-from-20002019"

def slug_from_dataset_item(item: object) -> str | None:
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
            got = slug_from_dataset_item(v) if v is not None else None
            if got:
                return got
    return None


def looks_like_dataset_slug(s: str) -> bool:
    if "/" not in s or s.count("/") != 1:
        return False
    owner, name = s.split("/", 1)
    if not owner or not name or owner.isdigit():
        return False
    return bool(re.match(r"^[\w-]+$", owner)) and bool(re.match(r"^[\w-]+$", name))


def walk_json_for_dataset_slugs(obj: object, out: list[str]) -> None:
    """Fallback: collect `owner/dataset` strings anywhere in the metadata JSON."""
    if isinstance(obj, dict):
        for v in obj.values():
            walk_json_for_dataset_slugs(v, out)
    elif isinstance(obj, list):
        for v in obj:
            walk_json_for_dataset_slugs(v, out)
    elif isinstance(obj, str) and looks_like_dataset_slug(obj.strip()):
        out.append(obj.strip())


def parse_dataset_slugs_from_kernel_metadata(metadata_path: str | os.PathLike[str]) -> list[str]:
    """Read `kernel-metadata.json` from `kernels_pull(..., metadata=True)` and extract dataset slugs."""
    mp = os.fspath(metadata_path)
    with open(mp, encoding="utf-8") as f:
        data = json.loads(f.read())
    slugs: list[str] = []
    for key in ("dataset_sources", "dataset_data_sources"):
        val = data.get(key)
        if not val:
            continue
        if isinstance(val, list):
            for item in val:
                s = slug_from_dataset_item(item)
                if s:
                    slugs.append(s)
        elif isinstance(val, str):
            s = slug_from_dataset_item(val)
            if s:
                slugs.append(s)

    if not slugs:
        extra: list[str] = []
        walk_json_for_dataset_slugs(data, extra)
        slugs = extra

    #this keeps order
    seen: set[str] = set()
    out: list[str] = []
    for s in slugs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def find_first_csv_under(root: str | os.PathLike[str]) -> str | None:
    root = os.fspath(root)
    matches: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                matches.append(os.path.join(dirpath, fn))
    matches.sort()
    return matches[0] if matches else None


def download_top_hits_spotify_via_kaggle_api(
    *,
    download_root: str | os.PathLike[str],
    kernel: str | None = None,
    dataset_fallback: str | None = None,
) -> str:
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError("Install the `kaggle` package") from e

    kernel = (kernel or TOP_HITS_SPOTIFY_KERNEL).strip()
    dataset_fallback = (dataset_fallback or TOP_HITS_SPOTIFY_DATASET_FALLBACK).strip()

    download_root = os.fspath(download_root)
    kernel_dir = os.path.join(download_root, "top_hits_spotify_kernel")
    os.makedirs(kernel_dir, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.kernels_pull(kernel, path=kernel_dir, metadata=True, quiet=True)

    meta_path = os.path.join(kernel_dir, "kernel-metadata.json")
    slugs: list[str] = []
    if os.path.isfile(meta_path):
        slugs = parse_dataset_slugs_from_kernel_metadata(meta_path)

    if not slugs:
        slugs = [dataset_fallback]

    data_root = os.path.join(download_root, "top_hits_spotify_data")
    os.makedirs(data_root, exist_ok=True)

    for slug in slugs:
        safe = slug.replace("/", "__")
        target = os.path.join(data_root, safe)
        os.makedirs(target, exist_ok=True)
        api.dataset_download_files(slug, path=target, unzip=True)

    csv_path = find_first_csv_under(data_root)
    if csv_path is None:
        raise FileNotFoundError(
            f"No .csv found under {data_root} after downloading: {slugs}. "
        )
    return csv_path