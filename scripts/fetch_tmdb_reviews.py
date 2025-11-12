"""
Fetch TMDB reviews for titles in netflix_titles.csv and append to data/reviews_raw.csv

Usage:
  python scripts/fetch_tmdb_reviews.py --netflix data/netflix_titles.csv --limit 200
"""

import argparse
import os
import pandas as pd
from api_tmdb import fetchReviewsForTitle

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
    args = ap.parse_args()

    if not os.path.exists(args.netflix):
        raise FileNotFoundError(f"Netflix CSV not found: {args.netflix}")

    nf = pd.read_csv(args.netflix)
    expected_cols = {"show_id","type","title","release_year","description"}
    missing = expected_cols - set(nf.columns)
    if missing:
        raise ValueError(f"Missing columns in netflix_titles.csv: {missing}")

    # keep minimal columns
    nf = nf[list(expected_cols)].copy()
    nf["type_norm"] = nf["type"].map(_normalizeType)
    nf["year_int"] = nf["release_year"].map(_safeInt)

    rows = []
    count = 0
    for _, row in nf.iterrows():
        if count >= args.limit:
            break
        title = str(row["title"])
        year = row["year_int"]
        kind = row["type_norm"]
        try:
            reviews = fetchReviewsForTitle(title=title, year=year, kind=kind)
            for r in reviews:
                rows.append({
                    "show_id": row["show_id"],
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
        except Exception as e:
            # skip; keep going
            rows.append({
                "show_id": row["show_id"],
                "type": row["type"],
                "title": title,
                "release_year": row["release_year"],
                "description": row["description"],
                "review_text": "",
                "author": None,
                "created_at": None,
                "url": None,
                "source": "tmdb_error"
            })
        count += 1

    new_df = pd.DataFrame(rows)
    if os.path.exists(args.output):
        old = pd.read_csv(args.output)
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df

    # basic dedupe on (show_id, review_text)
    out = out.dropna(subset=["review_text"])
    out["review_text_norm"] = out["review_text"].astype(str).str.strip()
    out = out.drop_duplicates(subset=["show_id", "review_text_norm"]).drop(columns=["review_text_norm"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")

if __name__ == "__main__":
    main()
