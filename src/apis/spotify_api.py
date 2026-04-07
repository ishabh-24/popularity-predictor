from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SpotifyConfig:

    client_id: str
    client_secret: str


class SpotifyClient:

    def __init__(self, cfg: SpotifyConfig):
        self.cfg = cfg
        self._sp = None

    def _get(self):
        if self._sp is not None:
            return self._sp

        try:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency for Spotify client. Install `spotipy`."
            ) from e

        auth_manager = SpotifyClientCredentials(
            client_id=self.cfg.client_id,
            client_secret=self.cfg.client_secret,
        )
        self._sp = spotipy.Spotify(auth_manager=auth_manager)
        return self._sp

    # -------- high level methods --------

    def search_track(self, *, track_name: str, artist_name: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        """
        Search Spotify for candidate tracks.
        Returns Spotipy track objects (raw), so downstream can choose best match.
        """
        q = f'track:"{track_name}"'
        if artist_name:
            q += f' artist:"{artist_name}"'

        sp = self._get()
        res = sp.search(q=q, type="track", limit=limit)
        return res.get("tracks", {}).get("items", [])

    def get_audio_features(self, track_ids: Iterable[str]) -> list[dict[str, Any]]:
        """
        Batch fetch audio features.
        Returns raw audio feature dicts (one per id, may include None for missing).
        """
        sp = self._get()
        ids = list(track_ids)
        if not ids:
            return []
        return sp.audio_features(ids) or []

    def get_track(self, track_id: str) -> dict[str, Any]:
        sp = self._get()
        return sp.track(track_id)

    def get_artist(self, artist_id: str) -> dict[str, Any]:
        sp = self._get()
        return sp.artist(artist_id)


def normalize_spotify_track(track: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a Spotipy track object into stable, model-friendly fields.
    """
    artists = track.get("artists") or []
    primary_artist = artists[0] if artists else {}
    album = track.get("album") or {}

    return {
        "spotify_track_id": track.get("id"),
        "track_name": track.get("name"),
        "artist_name": primary_artist.get("name"),
        "spotify_artist_id": primary_artist.get("id"),
        "release_date": album.get("release_date"),
        "spotify_popularity_track": track.get("popularity"),
    }


def normalize_spotify_audio_features(af: dict[str, Any] | None) -> dict[str, Any]:
    if not af:
        return {}

    keep = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]
    out = {k: af.get(k) for k in keep}
    out["spotify_track_id"] = af.get("id")
    return out


def normalize_spotify_artist(artist: dict[str, Any]) -> dict[str, Any]:
    return {
        "spotify_artist_id": artist.get("id"),
        "artist_name": artist.get("name"),
        "artist_popularity": artist.get("popularity"),
        "artist_followers": (artist.get("followers") or {}).get("total"),
        "genre": (artist.get("genres") or [None])[0],
    }

