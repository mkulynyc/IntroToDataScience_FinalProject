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

### Difference
def main(netflix_path="data/netflix_titles.csv", output_path="data/reviews_raw.csv", limit=200):
    import pandas as pd
    import os
    from api_tmdb import fetchReviewsForTitle

    if not os.path.exists(netflix_path):
        raise FileNotFoundError(f"Netflix CSV not found: {netflix_path}")

    nf = pd.read_csv(netflix_path)
    expected_cols = {"show_id", "type", "title", "release_year", "description"}
    missing = expected_cols - set(nf.columns)
    if missing:
        raise ValueError(f"Missing columns in netflix_titles.csv: {missing}")

    nf = nf[list(expected_cols)].copy()
    nf["type_norm"] = nf["type"].map(lambda t: str(t or "").strip().lower() if t else "movie")
    nf["year_int"] = nf["release_year"].map(lambda x: int(x) if pd.notna(x) and str(x).isdigit() else None)

    rows = []
    count = 0
    for _, row in nf.iterrows():
        if count >= limit:
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
        except Exception:
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
    if os.path.exists(output_path):
        old = pd.read_csv(output_path)
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df

    out = out.dropna(subset=["review_text"])
    out["review_text_norm"] = out["review_text"].astype(str).str.strip()
    out = out.drop_duplicates(subset=["show_id", "review_text_norm"]).drop(columns=["review_text_norm"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} rows to {output_path}")

if __name__ == "__main__":
    main()
