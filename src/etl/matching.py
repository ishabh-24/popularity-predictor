from __future__ import annotations

import re
from dataclasses import dataclass


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9\s]")
_FEAT = re.compile(r"\b(ft|feat|featuring)\b.*$", re.IGNORECASE)


def normalize_text(s: str | None) -> str:
    if not s:
        return ""
    s = s.lower()
    s = _FEAT.sub("", s)
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


@dataclass(frozen=True)
class MatchKey:
    track_norm: str
    artist_norm: str

    @staticmethod
    def from_row(track_name: str | None, artist_name: str | None) -> "MatchKey":
        return MatchKey(normalize_text(track_name), normalize_text(artist_name))


def simple_match_score(a: MatchKey, b: MatchKey) -> float:
    """
    Baseline deterministic match score for joining Billboard rows to Spotify search results.
    """
    if not a.track_norm or not b.track_norm:
        return 0.0
    track = 1.0 if a.track_norm == b.track_norm else (0.6 if a.track_norm in b.track_norm or b.track_norm in a.track_norm else 0.0)
    artist = 1.0 if (a.artist_norm and a.artist_norm == b.artist_norm) else (0.4 if a.artist_norm and b.artist_norm and (a.artist_norm in b.artist_norm or b.artist_norm in a.artist_norm) else 0.0)
    return 0.7 * track + 0.3 * artist

