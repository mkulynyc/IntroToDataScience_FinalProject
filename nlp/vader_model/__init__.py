from typing import Iterable, Optional
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

__all__ = ["applyVader", "vaderLabelFromCompound"]

_VADER = None

def _getVader():
    global _VADER
    if _VADER is None:
        _VADER = SentimentIntensityAnalyzer()
    return _VADER

def vaderLabelFromCompound(compound: float, posThresh: float = 0.05, negThresh: float = -0.05) -> str:
    if compound >= posThresh:
        return "POSITIVE"
    if compound <= negThresh:
        return "NEGATIVE"
    return "NEUTRAL"

def applyVader(
    df: pd.DataFrame,
    textCol: str = "text",
    posThresh: float = 0.05,
    negThresh: float = -0.05,
    outPrefix: str = "vader_",
    batchSize: Optional[int] = None,
) -> pd.DataFrame:
    """
    Adds columns:
      vader_pos, vader_neu, vader_neg, vader_compound, vader_label
    """
    sia = _getVader()
    texts: Iterable[str] = df[textCol].astype(str).fillna("")
    scores = [sia.polarity_scores(t) for t in texts]

    df[outPrefix + "pos"] = [s["pos"] for s in scores]
    df[outPrefix + "neu"] = [s["neu"] for s in scores]
    df[outPrefix + "neg"] = [s["neg"] for s in scores]
    df[outPrefix + "compound"] = [s["compound"] for s in scores]
    df[outPrefix + "label"] = [vaderLabelFromCompound(s["compound"], posThresh, negThresh) for s in scores]
    return df
