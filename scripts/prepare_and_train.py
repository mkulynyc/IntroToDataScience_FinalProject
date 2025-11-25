"""
Prepare reviews (clean + sample), weak-label with VADER, and run a quick spaCy textcat train.
Outputs:
- data/reviews_clean.csv
- data/label_sample.csv (id,text)  <-- for manual labeling
- data/train_reviews_weak.csv (text,label)
- data/test_reviews_weak.csv (text,label)
- nlp/spacy_model/artifacts/best  (saved spaCy model)
Run: python scripts/prepare_and_train.py --sample-size 400 --epochs 12
"""

import argparse
import os
import sys
import csv
from pathlib import Path
import random

import pandas as pd
from tqdm import tqdm

# VADER
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    print("Please install vaderSentiment (pip install vaderSentiment).", file=sys.stderr)
    raise

# spaCy
try:
    import spacy
    from spacy.util import minibatch
    # compounding scheduler lives in spacy.util.schedules for newer spaCy
    try:
        from spacy.util.schedules import compounding
    except Exception:
        # fallback for older spaCy versions
        from spacy.util import compounding
    from spacy.training import Example
except Exception:
    print("Please install spacy (pip install spacy).", file=sys.stderr)
    raise

def find_reviews_file():
    # Try common paths
    candidates = [
        Path("data/reviews_raw.csv"),
        Path("data/raw_reviews.csv"),
        Path("data/reviews.csv"),
        Path("data/reviews_raw_merged.csv")
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No reviews file found in {candidates}. Please place fetched reviews in one of those paths.")

def clean_reviews(df):
    # Ensure columns exist for 'review_text' or 'review' or 'text'
    col_candidates = ['review_text','review','text','description']
    text_col = None
    for c in col_candidates:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        for c in df.columns:
            if c.lower() in ['review_text','review','text','description']:
                text_col = c
                break
    if text_col is None:
        text_col = df.columns[-1]

    # Remove placeholder/error rows that contain 'tmdb_error' or 'Client Error' or 'Unauthorized' or '401'
    mask_error = df.apply(lambda row: row.astype(str).str.contains('tmdb_error|Client Error|Unauthorized|401|tmdb error', case=False, na=False).any(), axis=1)
    df_clean = df.loc[~mask_error].copy()
    df_clean = df_clean.rename(columns={text_col: 'text'})
    df_clean['text'] = df_clean['text'].astype(str).str.strip()
    df_clean = df_clean[df_clean['text'].str.len() > 5].copy()
    df_clean = df_clean.drop_duplicates(subset=['text']).reset_index(drop=True)
    return df_clean[['text'] + [c for c in df_clean.columns if c != 'text']]

def create_label_sample(df, sample_size=400, stratify_by=None, out_path=Path("data/label_sample.csv")):
    if stratify_by and stratify_by in df.columns:
        ids = []
        grouped = df.groupby(stratify_by)
        groups = list(grouped.groups.keys())
        random.shuffle(groups)
        for gid in groups:
            group_df = df[df[stratify_by] == gid]
            row = group_df.sample(n=1)
            ids.append(row.index[0])
            if len(ids) >= sample_size:
                break
        if len(ids) < sample_size:
            rem = sample_size - len(ids)
            remaining_idx = df.index.difference(ids)
            ids += list(remaining_idx.to_series().sample(n=min(rem, len(remaining_idx))))
        sample_df = df.loc[ids].copy()
    else:
        sample_df = df.sample(n=min(sample_size, len(df))).copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df_reset = sample_df.reset_index().rename(columns={'index':'id'})[['id','text']]
    sample_df_reset.to_csv(out_path, index=False)
    return out_path, sample_df_reset

def weak_label_vader(df, analyzer=None, pos_thresh=0.05, neg_thresh=-0.05):
    if analyzer is None:
        analyzer = SentimentIntensityAnalyzer()
    labels = []
    compounds = []
    for txt in tqdm(df['text'].astype(str), desc="VADER labeling"):
        s = analyzer.polarity_scores(txt)
        c = s['compound']
        compounds.append(c)
        if c >= pos_thresh:
            labels.append("POSITIVE")
        elif c <= neg_thresh:
            labels.append("NEGATIVE")
        else:
            labels.append(None)
    df2 = df.copy()
    df2['vader_compound'] = compounds
    df2['weak_label'] = labels
    df2 = df2.dropna(subset=['weak_label']).reset_index(drop=True)
    return df2

def train_spacy_textcat(train_df, dev_df, n_iter=12, model_output_dir=Path("nlp/spacy_model/artifacts/best")):
    nlp = spacy.blank("en")
    from typing import Any
    if "textcat" not in nlp.pipe_names:
        textcat: Any = nlp.add_pipe("textcat")
    else:
        textcat: Any = nlp.get_pipe("textcat")
    labels = train_df['label'].unique().tolist()
    for l in labels:
        textcat.add_label(l)
    # Prepare Example objects for spaCy v3+ training API
    train_examples = []
    for text, label in zip(train_df['text'], train_df['label']):
        cats = {lbl: (label == lbl) for lbl in labels}
        train_examples.append(Example.from_dict(nlp.make_doc(str(text)), {"cats": cats}))
    dev_examples = []
    for text, label in zip(dev_df['text'], dev_df['label']):
        cats = {lbl: (label == lbl) for lbl in labels}
        dev_examples.append(Example.from_dict(nlp.make_doc(str(text)), {"cats": cats}))

    # Initialize the pipeline (required to set up weights)
    nlp.initialize(lambda: train_examples)
    optimizer = nlp.create_optimizer()

    for epoch in range(n_iter):
        random.shuffle(train_examples)
        losses = {}
        batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.5))
        for batch in batches:
            nlp.update(batch, sgd=optimizer, losses=losses)
        # simple dev eval
        correct = 0
        total = 0
        for ex in dev_examples:
            doc = nlp(ex.reference.text)
            # choose highest-scoring category robustly
            predicted = max(doc.cats.items(), key=lambda kv: kv[1])[0] if getattr(doc, 'cats', None) else None
            gold = max(ex.reference.cats.items(), key=lambda kv: kv[1])[0] if ex.reference.cats else None
            if predicted == gold:
                correct += 1
            total += 1
        acc = correct / total if total else 0.0
        print(f"Epoch {epoch+1}/{n_iter} — loss: {losses.get('textcat',0):.4f} — dev_acc: {acc:.4f}")
    model_output_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(str(model_output_dir))
    print("Saved spaCy model to", model_output_dir)
    return model_output_dir

def main(args):
    random.seed(42)
    try:
        reviews_path = find_reviews_file()
        print("Found reviews file:", reviews_path)
    except Exception as e:
        print("ERROR: ", e)
        sys.exit(1)

    df_raw = pd.read_csv(reviews_path)
    print("Raw rows:", len(df_raw))

    df_clean = clean_reviews(df_raw)
    print("Clean rows (after removing errors/dupes/short):", len(df_clean))
    Path("data").mkdir(parents=True, exist_ok=True)
    df_clean.to_csv("data/reviews_clean.csv", index=False)
    print("Wrote data/reviews_clean.csv")

    sample_path, sample_df = create_label_sample(df_clean, sample_size=args.sample_size, stratify_by='show_id' if 'show_id' in df_clean.columns else None)
    print("Wrote labeling sample to", sample_path)

    analyzer = SentimentIntensityAnalyzer()
    df_weak = weak_label_vader(df_clean, analyzer=analyzer, pos_thresh=args.pos_thresh, neg_thresh=args.neg_thresh)
    print("Weak-labeled rows:", len(df_weak))

    df_weak = df_weak.rename(columns={'weak_label':'label'})
    label_counts = df_weak['label'].value_counts().to_dict()
    print("Label counts:", label_counts)
    if len(df_weak) < 50:
        print("Warning: too few weak-labeled examples (<50). Consider increasing fetch or allowing neutral labels.")
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df_weak[['text','label']], test_size=0.2, stratify=df_weak['label'], random_state=42)
    train_df.to_csv("data/train_reviews_weak.csv", index=False)
    test_df.to_csv("data/test_reviews_weak.csv", index=False)
    print("Wrote data/train_reviews_weak.csv and data/test_reviews_weak.csv")
    model_dir = train_spacy_textcat(train_df, test_df, n_iter=args.epochs, model_output_dir=Path("nlp/spacy_model/artifacts/best"))
    print("\nSummary:")
    print("- cleaned:", "data/reviews_clean.csv")
    print("- sample:", sample_path)
    print("- weak train/test:", "data/train_reviews_weak.csv", "data/test_reviews_weak.csv")
    print("- spaCy model:", model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=400, help="Number of rows to sample for manual labeling")
    parser.add_argument("--pos-thresh", type=float, default=0.05, help="VADER positive threshold (compound >= this => POSITIVE)")
    parser.add_argument("--neg-thresh", type=float, default=-0.05, help="VADER negative threshold (compound <= this => NEGATIVE)")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs for the quick spaCy train")
    args = parser.parse_args()
    main(args)
