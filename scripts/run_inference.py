import argparse
import pandas as pd

from nlp.utils import cleanText
from nlp.vader_model import applyVader
from nlp.spacy_model import applySpacy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with at least a text column")
    ap.add_argument("--textCol", default="text")
    ap.add_argument("--spacyModel", default="nlp/spacy_model/artifacts/best")
    ap.add_argument("--output", required=True, help="Output CSV path")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.textCol not in df.columns:
        raise ValueError(f"Column {args.textCol!r} not found in {args.input}")

    df[args.textCol] = df[args.textCol].fillna("").astype(str).map(cleanText)

    df = applyVader(df, textCol=args.textCol)
    df = applySpacy(df, textCol=args.textCol, modelPath=args.spacyModel)

    df.to_csv(args.output, index=False)
    print(f"Wrote: {args.output} with VADER and spaCy scores")

if __name__ == "__main__":
    main()
