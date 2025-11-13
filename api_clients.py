import os
import requests
from typing import List, Dict, Optional

TMDB_BASE = "https://api.themoviedb.org/3"

def _getTmdbKey(streamlit_secrets=None) -> str:
    if streamlit_secrets and "TMDB_API_KEY" in streamlit_secrets:
        return streamlit_secrets["TMDB_API_KEY"]
    return os.getenv("TMDB_API_KEY", "")

def tmdbSearch(title: str, year: Optional[int], isTv: bool, apiKey: str) -> Optional[Dict]:
    endpoint = f"{TMDB_BASE}/{'search/tv' if isTv else 'search/movie'}"
    params = {"api_key": apiKey, "query": title, "page": 1, "include_adult": "false"}
    if year and not isTv:
        params["year"] = year
    if year and isTv:
        params["first_air_date_year"] = year
    r = requests.get(endpoint, params=params, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])
    return results[0] if results else None

def tmdbReviews(tmdbId: int, isTv: bool, apiKey: str, maxPages: int = 2) -> List[Dict]:
    coll: List[Dict] = []
    page = 1
    while page <= maxPages:
        endpoint = f"{TMDB_BASE}/{'tv' if isTv else 'movie'}/{tmdbId}/reviews"
        params = {"api_key": apiKey, "page": page}
        r = requests.get(endpoint, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for rev in data.get("results", []):
            coll.append({
                "author": rev.get("author") or "",
                "review_text": rev.get("content") or "",
                "review_date": (rev.get("created_at") or "")[:10],
                "url": rev.get("url") or "",
                "source": "tmdb",
                "rating_10": None
            })
        if page >= (data.get("total_pages") or 1):
            break
        page += 1
    return coll

def fetchTmdbReviewsForTitle(title: str, year: Optional[int], isTv: bool, streamlit_secrets=None, maxPages: int = 2) -> List[Dict]:
    apiKey = _getTmdbKey(streamlit_secrets=streamlit_secrets)
    if not apiKey:
        raise RuntimeError("TMDB_API_KEY not found (set in .streamlit/secrets.toml or environment).")
    match = tmdbSearch(title=title, year=year, isTv=isTv, apiKey=apiKey)
    if not match:
        return []
    tmdbId = match["id"]
    return tmdbReviews(tmdbId=tmdbId, isTv=isTv, apiKey=apiKey, maxPages=maxPages)

