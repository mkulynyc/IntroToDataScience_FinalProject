"""
Merge netflix_titles.csv + reviews_raw.csv (fetched) into a per-title enriched text field,
then run VADER on the merged data and write to data/netflix_enriched_vader.csv
"""
import os
import pandas as pd
from nlp.utils import cleanText
from nlp.vader_model import applyVader


def buildTextColumn(netflix: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    # aggregate reviews per show
    agg = (reviews
           .dropna(subset=["review_text"]) 
           .assign(review_text=lambda d: d["review_text"].astype(str).str.strip())
           .groupby("show_id", as_index=False)
           .agg(review_join=("review_text", lambda s: " || ".join(s.tolist()))))

    merged = netflix.merge(agg, on="show_id", how="left")
    merged["nlp_text"] = (merged["description"].fillna("").astype(str).str.strip() + " " +
                           merged["review_join"].fillna("").astype(str).str.strip()).str.strip()
    merged["nlp_text"] = merged.apply(lambda r: r["description"] if not r["nlp_text"] else r["nlp_text"], axis=1)
    return merged


def main():
    netflix_path = "data/netflix_titles.csv"
    reviews_path = "data/reviews_raw.csv"
    out_path = "data/netflix_enriched_vader.csv"

    if not os.path.exists(netflix_path):
        raise FileNotFoundError(netflix_path)
    if not os.path.exists(reviews_path):
        raise FileNotFoundError(reviews_path)

    nf = pd.read_csv(netflix_path)
    rv = pd.read_csv(reviews_path)

    needed = {"show_id","type","title","release_year","description"}
    missing = needed - set(nf.columns)
    if missing:
        raise ValueError(f"Missing columns in netflix_titles.csv: {missing}")

    df = buildTextColumn(nf, rv)
    df["nlp_text"] = df["nlp_text"].fillna("").astype(str).map(cleanText)

    df = applyVader(df, textCol="nlp_text")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote enriched + VADER-scored dataset to {out_path} (rows={len(df)})")


if __name__ == "__main__":
    main()
