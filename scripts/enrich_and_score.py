"""
Merge netflix_titles.csv + reviews_raw.csv, create a single text field, run VADER + spaCy,
and save to data/netflix_enriched_scored.csv

Usage:
  python scripts/enrich_and_score.py --netflix data/netflix_titles.csv --reviews data/reviews_raw.csv
"""

import argparse
import os
import pandas as pd

from nlp.utils import cleanText
from nlp.vader_model import applyVader
from nlp.spacy_model import applySpacy

def buildTextColumn(netflix: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates multiple reviews per show into one joined text field alongside the description.
    """
    # aggregate reviews per show
    agg = (reviews
           .dropna(subset=["review_text"])
           .assign(review_text=lambda d: d["review_text"].astype(str).str.strip())
           .groupby("show_id", as_index=False)
           .agg(review_join=("review_text", lambda s: " || ".join(s.tolist()))))

    merged = netflix.merge(agg, on="show_id", how="left")
    # concatenate description + reviews
    merged["nlp_text"] = (merged["description"].fillna("").astype(str).str.strip() + " " +
                          merged["review_join"].fillna("").astype(str).str.strip()).str.strip()

    # fallback: if empty, just use description
    merged["nlp_text"] = merged.apply(
        lambda r: r["description"] if not r["nlp_text"] else r["nlp_text"], axis=1
    )
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netflix", default="data/netflix_titles.csv")
    ap.add_argument("--reviews", default="data/reviews_raw.csv")
    ap.add_argument("--output", default="data/netflix_enriched_scored.csv")
    ap.add_argument("--spacyModel", default="nlp/spacy_model/artifacts/best")
    args = ap.parse_args()

    if not os.path.exists(args.netflix):
        raise FileNotFoundError(f"Netflix CSV not found: {args.netflix}")
    if not os.path.exists(args.reviews):
        raise FileNotFoundError(f"Reviews CSV not found: {args.reviews}")

    nf = pd.read_csv(args.netflix)
    rv = pd.read_csv(args.reviews)
    needed = {"show_id","type","title","release_year","description"}
    missing = needed - set(nf.columns)
    if missing:
        raise ValueError(f"Missing columns in netflix_titles.csv: {missing}")

    # build text column
    df = buildTextColumn(nf, rv)
    df["nlp_text"] = df["nlp_text"].fillna("").astype(str).map(cleanText)

    # run VADER + spaCy
    df = applyVader(df, textCol="nlp_text")
    df = applySpacy(df, textCol="nlp_text", modelPath=args.spacyModel)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote enriched + scored dataset to {args.output} (rows={len(df)})")

if __name__ == "__main__":
    main()
