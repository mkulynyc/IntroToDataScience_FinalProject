"""
Run VADER sentiment on netflix_titles.csv and write results
"""
import argparse
import pandas as pd
from nlp.vader_model import applyVader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/netflix_titles.csv')
    ap.add_argument('--output', default='data/netflix_vader_scored.csv')
    ap.add_argument('--textCol', default='description')
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.textCol not in df.columns:
        raise ValueError(f"Column {args.textCol!r} not in {args.input}")
    df['text'] = df[args.textCol].fillna('').astype(str)
    df = applyVader(df, textCol='text')
    out_cols = ['show_id','title', args.textCol, 'vader_pos','vader_neu','vader_neg','vader_compound','vader_label']
    out_cols = [c for c in out_cols if c in df.columns]
    df.to_csv(args.output, index=False, columns=out_cols)
    print(f"Wrote: {args.output}")

if __name__ == '__main__':
    main()
