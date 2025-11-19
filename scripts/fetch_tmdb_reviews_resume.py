"""
Resume-capable TMDB review fetcher.

Usage examples:
  # Fetch next 500 titles (skipping already fetched) and append to data/reviews_raw.csv
  python -m scripts.fetch_tmdb_reviews_resume --netflix data/netflix_titles.csv --output data/reviews_raw.csv --limit 500

  # Fetch with a small delay between requests to be polite
  python -m scripts.fetch_tmdb_reviews_resume --limit 200 --batchDelay 0.2

This script will:
 - read existing output (if present) and skip show_ids already fetched
 - append new reviews incrementally to the output file so progress is saved
 - support a start index and limit for batch runs
 - use the same TMDB credentials as the other scripts (.streamlit/secrets.toml or env vars)
"""

import argparse
import os
import pandas as pd
import time
import requests
from tqdm import tqdm
from pandas.errors import EmptyDataError

# TMDB credentials: first check environment, then rely on .streamlit/secrets.toml
V3 = os.getenv("TMDB_API_KEY")
V4 = os.getenv("TMDB_ACCESS_READ_TOKEN")

S = requests.Session()
S.headers.update({"Accept": "application/json"})
if V4:
    S.headers.update({"Authorization": f"Bearer {V4}"})


def tmdb_search(kind: str, query: str, year=None):
    url = f"https://api.themoviedb.org/3/search/{kind}"
    params = {"query": query}
    if not V4 and V3:
        params["api_key"] = V3
    if year and kind == "movie":
        params["year"] = str(int(year))
    elif year and kind == "tv":
        params["first_air_date_year"] = str(int(year))

    r = S.get(url, params=params, timeout=12)
    r.raise_for_status()
    return r.json().get("results", [])


def fetchReviewsForTitle(title: str, year=None, kind="movie"):
    results = tmdb_search(kind, title, year)
    if not results:
        return []

    movie_id = results[0]["id"]
    url = f"https://api.themoviedb.org/3/{kind}/{movie_id}/reviews"
    params = {} if V4 else {"api_key": V3}

    r = S.get(url, params=params, timeout=12)
    r.raise_for_status()

    data = r.json()
    reviews = []
    for rev in data.get("results", []):
        reviews.append({
            "content": rev.get("content", ""),
            "author": rev.get("author"),
            "created_at": rev.get("created_at"),
            "url": rev.get("url"),
            "source": "tmdb"
        })
    return reviews


def _normalizeType(t: str) -> str:
    s = str(t or "").strip().lower()
    if s in {"movie", "movies", "film"}:
        return "movie"
    if s in {"tv show", "tv", "show", "series"}:
        return "tv"
    return "movie"


def _safeInt(x):
    try:
        return int(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netflix", default="data/netflix_titles.csv", help="Path to netflix_titles.csv")
    ap.add_argument("--output", default="data/reviews_raw.csv", help="Path to append/save reviews CSV")
    ap.add_argument("--limit", type=int, default=200, help="Limit number of titles to fetch in this run")
    ap.add_argument("--start", type=int, default=0, help="Start index in netflix file (0-based)")
    ap.add_argument("--batchDelay", type=float, default=0.1, help="Delay (s) between requests to TMDB to be polite")
    ap.add_argument("--retries", type=int, default=2, help="Number of retries for failed requests")
    args = ap.parse_args()

    if not os.path.exists(args.netflix):
        raise FileNotFoundError(f"Netflix CSV not found: {args.netflix}")

    nf = pd.read_csv(args.netflix)
    expected_cols = {"show_id", "type", "title", "release_year", "description"}
    missing = expected_cols - set(nf.columns)
    if missing:
        raise ValueError(f"Missing columns in netflix_titles.csv: {missing}")

    nf = nf[list(expected_cols)].copy()
    nf["type_norm"] = nf["type"].map(_normalizeType)
    nf["year_int"] = nf["release_year"].map(_safeInt)

    # Load existing output to skip already fetched show_ids
    if os.path.exists(args.output):
        try:
            old = pd.read_csv(args.output)
            fetched_show_ids = set(old["show_id"].astype(str).tolist()) if "show_id" in old.columns else set()
        except EmptyDataError:
            old = pd.DataFrame()
            fetched_show_ids = set()
    else:
        old = pd.DataFrame()
        fetched_show_ids = set()

    rows = []
    count = 0
    start_idx = args.start
    total = len(nf)

    # iterate with tqdm for visibility
    it = list(nf.iterrows())[start_idx:]
    for i, row in tqdm(it, total=min(args.limit, total - start_idx), desc="Fetching reviews"):
        if count >= args.limit:
            break

        sid = str(row["show_id"])
        if sid in fetched_show_ids:
            # skip already fetched
            start_idx += 1
            continue

        title = str(row["title"])
        year = row["year_int"]
        kind = row["type_norm"]

        success = False
        attempt = 0
        while not success and attempt <= args.retries:
            try:
                reviews = fetchReviewsForTitle(title=title, year=year, kind=kind)
                for r in reviews:
                    rows.append({
                        "show_id": sid,
                        "type": row["type"],
                        "title": title,
                        "release_year": row["release_year"],
                        "description": row["description"],
                        "review_text": r["content"],
                        "author": r.get("author"),
                        "created_at": r.get("created_at"),
                        "url": r.get("url"),
                        "source": r.get("source", "tmdb")
                    })
                success = True
            except Exception as e:
                attempt += 1
                if attempt > args.retries:
                    # record a placeholder row indicating error
                    rows.append({
                        "show_id": sid,
                        "type": row["type"],
                        "title": title,
                        "release_year": row["release_year"],
                        "description": row["description"],
                        "review_text": "",
                        "author": None,
                        "created_at": None,
                        "url": None,
                        "source": f"tmdb_error:{e}"
                    })
                else:
                    time.sleep(1.0 * attempt)
        # increment counters
        count += 1
        time.sleep(args.batchDelay)

        # Periodically flush to disk every 50 titles fetched
        if len(rows) >= 50:
            flush_rows(rows, args.output, old)
            old = pd.read_csv(args.output) if os.path.exists(args.output) else old
            rows = []

    # final flush
    if rows:
        flush_rows(rows, args.output, old)

    # done
    print(f"Appended new reviews to {args.output}")


def flush_rows(rows, output_path, old_df):
    new_df = pd.DataFrame(rows)
    if os.path.exists(output_path) and not old_df.empty:
        out = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out = new_df

    # remove duplicates
    out = out.dropna(subset=["review_text"]) if "review_text" in out.columns else out
    if "review_text" in out.columns:
        out["review_text_norm"] = out["review_text"].astype(str).str.strip()
        out = out.drop_duplicates(subset=["show_id", "review_text_norm"]).drop(columns=["review_text_norm"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Flushed {len(new_df)} new rows (total={len(out)}) to {output_path}")


if __name__ == "__main__":
    main()
