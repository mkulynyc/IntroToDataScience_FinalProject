from __future__ import annotations

import os
from typing import List, Dict
import pandas as pd
import spacy
from spacy.language import Language
from spacy.util import minibatch
from spacy.training.example import Example
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

__all__ = [
    "trainSpacyTextcat",
    "applySpacy",
    "evaluateSpacy",
    "prepareDataFrameLabels",
]

# ---------- label utils ----------

_ACCEPTED_POS = {"positive", "pos", "1", "true", "t", "yes", "y", "p"}
_ACCEPTED_NEG = {"negative", "neg", "0", "false", "f", "no", "n"}

def _normalizeLabel(x: str) -> str:
    if x is None:
        return "NEGATIVE"
    s = str(x).strip().lower()
    if s in _ACCEPTED_POS or s == "positive":
        return "POSITIVE"
    if s in _ACCEPTED_NEG or s == "negative":
        return "NEGATIVE"
    raise ValueError(f"Unrecognized label value: {x!r}. Use POSITIVE/NEGATIVE, pos/neg, 1/0, true/false.")

def prepareDataFrameLabels(df: pd.DataFrame, labelCol: str = "label") -> pd.DataFrame:
    y = df[labelCol].apply(_normalizeLabel)
    out = df.copy()
    out[labelCol] = y
    return out

def _dfToSpacyExamples(df: pd.DataFrame, textCol: str, labelCol: str, nlp: Language) -> List[Example]:
    exs: List[Example] = []
    for _, row in df.iterrows():
        text = str(row[textCol])
        label = str(row[labelCol]).upper()
        doc = nlp.make_doc(text)
        cats = {"POSITIVE": float(label == "POSITIVE"), "NEGATIVE": float(label == "NEGATIVE")}
        exs.append(Example.from_dict(doc, {"cats": cats}))
    return exs

# ---------- training / inference ----------

def trainSpacyTextcat(
    trainCsvPath: str,
    textCol: str = "text",
    labelCol: str = "label",
    nEpochs: int = 10,
    lr: float = 2e-3,
    dropout: float = 0.2,
    batchSize: int = 64,
    outputDir: str = "nlp/spacy_model/artifacts",
    devCsvPath: str | None = None,
) -> str:
    """
    Trains a spaCy textcat model on a blank English pipeline.
    Saves best model to <outputDir>/best and returns that path.
    """
    os.makedirs(outputDir, exist_ok=True)

    train_df = pd.read_csv(trainCsvPath)
    if devCsvPath:
        dev_df = pd.read_csv(devCsvPath)
    else:
        train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        cut = max(1, int(0.9 * len(train_df)))
        dev_df = train_df.iloc[cut:].reset_index(drop=True)
        train_df = train_df.iloc[:cut].reset_index(drop=True)

    train_df = prepareDataFrameLabels(train_df, labelCol)
    dev_df = prepareDataFrameLabels(dev_df, labelCol)

    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat")
    textcat.add_label("POSITIVE")
    textcat.add_label("NEGATIVE")

    train_examples = _dfToSpacyExamples(train_df, textCol, labelCol, nlp)
    dev_examples = _dfToSpacyExamples(dev_df, textCol, labelCol, nlp)

    optimizer = nlp.initialize(lambda: train_examples)

    best_f1 = -1.0
    best_path = os.path.join(outputDir, "best")

    for epoch in range(1, nEpochs + 1):
        losses = {}
        for batch in minibatch(train_examples, size=batchSize):
            nlp.update(batch, drop=dropout, sgd=optimizer, losses=losses, learn_rate=lr)

        # evaluate on dev
        y_true, y_pred = [], []
        for eg in dev_examples:
            doc = nlp(eg.reference.text)
            cats = doc.cats
            pred = "POSITIVE" if cats.get("POSITIVE", 0.0) >= cats.get("NEGATIVE", 0.0) else "NEGATIVE"
            gold = "POSITIVE" if eg.reference.cats.get("POSITIVE", 0.0) > 0.5 else "NEGATIVE"
            y_true.append(gold)
            y_pred.append(pred)

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        print(f"Epoch {epoch:02d} | loss={losses.get('textcat', 0):.4f} | macro-F1={f1:.4f} (P={p:.4f}, R={r:.4f})")

        if f1 > best_f1:
            best_f1 = f1
            nlp.to_disk(best_path)

    print(f"Best model saved to: {best_path} (macro-F1={best_f1:.4f})")
    return best_path

def applySpacy(
    df: pd.DataFrame,
    textCol: str = "text",
    modelPath: str = "nlp/spacy_model/artifacts/best",
    outPrefix: str = "spacy_",
) -> pd.DataFrame:
    if not os.path.exists(modelPath):
        raise FileNotFoundError(f"spaCy model not found at {modelPath}. Train it first.")
    nlp = spacy.load(modelPath)

    pos_probs, neg_probs, labels = [], [], []
    for t in df[textCol].astype(str).fillna(""):
        doc = nlp(t)
        pos = float(doc.cats.get("POSITIVE", 0.0))
        neg = float(doc.cats.get("NEGATIVE", 0.0))
        label = "POSITIVE" if pos >= neg else "NEGATIVE"
        pos_probs.append(pos)
        neg_probs.append(neg)
        labels.append(label)

    df[outPrefix + "pos_prob"] = pos_probs
    df[outPrefix + "neg_prob"] = neg_probs
    df[outPrefix + "label"] = labels
    return df

def evaluateSpacy(
    testCsvPath: str,
    textCol: str = "text",
    labelCol: str = "label",
    modelPath: str = "nlp/spacy_model/artifacts/best",
) -> Dict[str, object]:
    df = pd.read_csv(testCsvPath)
    df = prepareDataFrameLabels(df, labelCol)
    nlp = spacy.load(modelPath)

    y_true, y_pred = [], []
    for t, y in zip(df[textCol].astype(str), df[labelCol]):
        doc = nlp(t)
        pred = "POSITIVE" if doc.cats.get("POSITIVE", 0.0) >= doc.cats.get("NEGATIVE", 0.0) else "NEGATIVE"
        y_true.append(str(y))
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=["NEGATIVE", "POSITIVE"]).tolist()
    return {
        "accuracy": float(acc),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "confusion_matrix_labels": ["NEGATIVE", "POSITIVE"],
        "confusion_matrix": cm,
        "n": int(len(df)),
    }
