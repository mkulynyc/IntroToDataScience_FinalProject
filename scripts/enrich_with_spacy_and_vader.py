"""
Merge `data/netflix_titles.csv` with `data/reviews_raw.csv`, build a per-title `nlp_text`,
clean it, run VADER sentiment, run the saved spaCy textcat model (best_quick if present),
and write `data/netflix_enriched_final.csv` with both sets of scores.

This is intended to be a fast enrichment step using the weakly-trained quick model.
"""
import os
import sys
import pandas as pd
# ensure repo root is on sys.path so local `nlp` package can be imported
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nlp.utils.text_cleaning import cleanText
from nlp.vader_model import applyVader, vaderLabelFromCompound
import json


def buildTextColumn(netflix: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
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


def apply_spacy(nlp, texts, batch_size=64):
    """Run the spaCy pipeline over `texts` and return list of dicts for cats."""
    out = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        # doc.cats is a dict like {'POSITIVE':0.98, 'NEGATIVE':0.02}
        out.append(dict(doc.cats))
    return out


def main():
    netflix_path = "data/netflix_titles.csv"
    reviews_path = "data/reviews_raw.csv"
    out_path = "data/netflix_enriched_final.csv"

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

    # VADER
    df = applyVader(df, textCol="nlp_text")

    # spaCy
    spacy_model_paths = [
        "nlp/spacy_model/artifacts/best_quick",
        "nlp/spacy_model/artifacts/best",
    ]
    spacy_model = None
    for p in spacy_model_paths:
        if os.path.exists(p):
            spacy_model = p
            break

    if spacy_model is None:
        print("Warning: no spaCy model found at expected paths. Skipping spaCy scoring.")
        df.to_csv(out_path, index=False)
        print(f"Wrote enriched dataset with VADER only to {out_path} (rows={len(df)})")
        return

    import spacy
    nlp = spacy.load(spacy_model)

    cats = apply_spacy(nlp, df["nlp_text"].astype(str).fillna(""))
    # cats is list of dicts; convert to DataFrame (keys should be consistent)
    cats_df = pd.DataFrame(cats).fillna(0.0)
    # rename columns
    cats_df = cats_df.rename(columns={c: f"spacy_{c.lower()}" for c in cats_df.columns})

    # choose label by argmax probability
    def spacy_label(row):
        if row.empty:
            return "NEUTRAL"
        # find highest
        k = row.idxmax()
        # k is like 'spacy_positive' -> map to POSITIVE
        return k.replace('spacy_','').upper()

    out = pd.concat([df.reset_index(drop=True), cats_df.reset_index(drop=True)], axis=1)
    out["spacy_label"] = out[[c for c in out.columns if c.startswith('spacy_')]].apply(spacy_label, axis=1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    # Print summary
    print(f"Wrote enriched + VADER + spaCy dataset to {out_path} (rows={len(out)})")
    print("VADER label counts:")
    print(out["vader_label"].value_counts().to_dict())
    print("spaCy label counts:")
    print(out["spacy_label"].value_counts().to_dict())
    print("Sample rows:")
    print(out[["show_id","title","vader_label","vader_compound","spacy_label"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
