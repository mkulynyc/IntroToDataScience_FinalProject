"""
Generate a synthetic `data/netflix_enriched_scored.csv` by sampling
rows from `data/netflix_titles.csv` and adding plausible sentiment
columns so the Streamlit app can render many titles for testing.

Usage:
  python scripts/generate_enriched_sample.py --rows 200

This script is safe to run locally and deterministic (uses hashing for
pseudo-random but repeatable scores).
"""
import argparse
import hashlib
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
NETFLIX = os.path.join(ROOT, 'data', 'netflix_titles.csv')
OUT = os.path.join(ROOT, 'data', 'netflix_enriched_scored.csv')


def hash_to_float(s: str, a=0.0, b=1.0) -> float:
    # deterministically map string to float in [a,b)
    h = hashlib.md5(s.encode('utf-8')).hexdigest()
    x = int(h[:8], 16) / float(0xFFFFFFFF)
    return a + (b-a) * x


def vader_label_from_compound(c: float) -> str:
    if c >= 0.05:
        return 'POSITIVE'
    if c <= -0.05:
        return 'NEGATIVE'
    return 'NEUTRAL'


def spacy_label_from_prob(p: float) -> str:
    return 'POSITIVE' if p >= 0.5 else 'NEGATIVE'


def main(rows: int):
    if not os.path.exists(NETFLIX):
        print(f"Missing source file: {NETFLIX}")
        return
    df = pd.read_csv(NETFLIX)
    if df.empty:
        print("Source netflix_titles.csv is empty")
        return

    # sample up to `rows` rows (keep order deterministic)
    src = df.copy()
    src = src.reset_index(drop=True)
    n = min(rows, len(src))
    src = src.head(n)

    out = pd.DataFrame()
    out['title'] = src['title']
    out['type'] = src.get('type', 'Movie')
    out['release_year'] = src.get('release_year')
    out['nlp_text'] = src.get('description').fillna('')
    out['listed_in'] = src.get('listed_in').fillna('')

    # create deterministic pseudo-random sentiment scores from title
    vader_compound = []
    vader_label = []
    spacy_prob = []
    spacy_label = []
    review_join = []
    for t in out['title'].astype(str):
        v = hash_to_float(t, a=-1.0, b=1.0)
        vader_compound.append(v)
        vader_label.append(vader_label_from_compound(v))
        p = hash_to_float('spacy_'+t, a=0.0, b=1.0)
        spacy_prob.append(p)
        spacy_label.append(spacy_label_from_prob(p))
        # small synthetic review string (optional)
        review_join.append('' if p < 0.2 else 'Great movie || Loved it' if p > 0.8 else 'Interesting plot')

    out['vader_compound'] = vader_compound
    out['vader_label'] = vader_label
    out['spacy_pos_prob'] = spacy_prob
    out['spacy_label'] = spacy_label
    out['review_join'] = review_join

    out.to_csv(OUT, index=False)
    print(f"Wrote synthetic enriched file: {OUT} ({len(out)} rows)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=200, help='Number of rows to generate')
    args = parser.parse_args()
    main(args.rows)
