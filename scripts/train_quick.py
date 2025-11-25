"""
Quick trainer: load weak-labeled CSVs and run a short spaCy v3 textcat training on a capped subset.
Saves model to `nlp/spacy_model/artifacts/best_quick` and prints classification report on the test set.

Usage: python scripts/train_quick.py --max-train 2000 --epochs 6
"""
import argparse
from pathlib import Path
import random
import pandas as pd
import spacy
from spacy.util import minibatch
try:
    from spacy.util.schedules import compounding
except Exception:
    from spacy.util import compounding
from spacy.training import Example
from sklearn.metrics import classification_report


def load_data(train_path, test_path, max_train=None):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if max_train and len(train) > max_train:
        train = train.sample(n=max_train, random_state=42)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def train_quick(train_df, test_df, epochs=6, out_dir=Path('nlp/spacy_model/artifacts/best_quick')):
    labels = train_df['label'].unique().tolist()
    from typing import Any
    nlp = spacy.blank('en')
    textcat: Any = nlp.add_pipe('textcat')
    for l in labels:
        textcat.add_label(l)

    train_examples = [Example.from_dict(nlp.make_doc(str(t)), { 'cats': {lbl: (lbl==l) for lbl in labels} }) for t, l in zip(train_df['text'], train_df['label'])]
    dev_examples = [Example.from_dict(nlp.make_doc(str(t)), { 'cats': {lbl: (lbl==l) for lbl in labels} }) for t, l in zip(test_df['text'], test_df['label'])]

    nlp.initialize(lambda: train_examples)
    optimizer = nlp.create_optimizer()

    for epoch in range(epochs):
        random.shuffle(train_examples)
        losses = {}
        for batch in minibatch(train_examples, size=compounding(4.0, 16.0, 1.5)):
            nlp.update(batch, sgd=optimizer, losses=losses)
        # evaluate
        y_true = []
        y_pred = []
        for ex in dev_examples:
            doc = nlp(ex.reference.text)
            pred = max(doc.cats.items(), key=lambda kv: kv[1])[0] if getattr(doc, 'cats', None) else None
            gold = max(ex.reference.cats.items(), key=lambda kv: kv[1])[0] if ex.reference.cats else None
            y_true.append(gold)
            y_pred.append(pred)
        print(f'Epoch {epoch+1}/{epochs} — loss: {losses.get("textcat",0):.4f}')
        print(classification_report(y_true, y_pred, digits=4))

    out_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(str(out_dir))
    print('Saved quick model to', out_dir)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-train', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=6)
    args = parser.parse_args()

    train_path = Path('data/train_reviews_weak.csv')
    test_path = Path('data/test_reviews_weak.csv')
    if not train_path.exists() or not test_path.exists():
        print('Missing weak-labeled CSVs. Run prepare_and_train.py first to generate them.')
        return

    train_df, test_df = load_data(train_path, test_path, max_train=args.max_train)
    print('Train rows:', len(train_df), 'Test rows:', len(test_df))
    train_quick(train_df, test_df, epochs=args.epochs)


if __name__ == '__main__':
    main()
