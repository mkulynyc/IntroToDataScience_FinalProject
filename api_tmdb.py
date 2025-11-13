"""
TMDB reviews client for movies and TV.
Reads API key from .streamlit/secrets.toml ([tmdb].api_key) or env TMDB_API_KEY.
"""

import os
import time
from typing import Dict, List, Optional, Tuple
import requests

TMDB_BASE = "https://api.themoviedb.org/3"
_RATE_DELAY = 0.25  # seconds between calls to be nice

def _getApiKey() -> str:
    # streamlit secrets if available
    try:
        import streamlit as st  # will work when app runs; scripts ignore if no streamlit
        key = st.secrets.get("tmdb", {}).get("api_key")
        if key:
            return key
    except Exception:
        pass
    # env fallback
    key = os.getenv("TMDB_API_KEY")
    if not key:
        raise RuntimeError("TMDB API key not found. Set .streamlit/secrets.toml [tmdb].api_key or TMDB_API_KEY env.")
    return key

def _get(path: str, params: Dict) -> Dict:
    time.sleep(_RATE_DELAY)
    key = _getApiKey()
    params = dict(params or {})
    params["api_key"] = key
    r = requests.get(f"{TMDB_BASE}{path}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def searchTitle(title: str, year: Optional[int], kind: str) -> Optional[Tuple[str, int]]:
    """
    kind: 'movie' or 'tv'
    Returns (media_type, tmdb_id) or None
    """
    if kind not in {"movie", "tv"}:
        return None
    path = f"/search/{kind}"
    params = {"query": title}
    if year:
        params["year" if kind == "movie" else "first_air_date_year"] = year
    data = _get(path, params)
    results = data.get("results") or []
    if not results:
        return None
    return (kind, int(results[0]["id"]))

def getReviews(kind: str, tmdbId: int, maxPages: int = 2) -> List[Dict]:
    """
    Returns list of {'author','content','created_at','url','source'}.
    """
    out: List[Dict] = []
    for page in range(1, maxPages + 1):
        path = f"/{ 'movie' if kind=='movie' else 'tv' }/{tmdbId}/reviews"
        data = _get(path, {"page": page})
        for r in data.get("results", []):
            out.append({
                "author": r.get("author"),
                "content": r.get("content") or "",
                "created_at": r.get("created_at"),
                "url": r.get("url"),
                "source": "tmdb"
            })
        if page >= (data.get("total_pages") or 1):
            break
    return out

def fetchReviewsForTitle(title: str, year: Optional[int], kind: str) -> List[Dict]:
    """
    Search and fetch up to a couple pages of reviews for one title.
    """
    found = searchTitle(title, year, kind)
    if not found:
        return []
    _, tmdb_id = found
    return getReviews(kind, tmdb_id)
