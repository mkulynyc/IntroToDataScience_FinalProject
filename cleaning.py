"""
Generic text dataframe cleaner used by the NLP pipelines.

Functions assume a dataframe with a text column (default: 'text').
"""

import pandas as pd
from nlp.utils import cleanText

__all__ = ["loadAndCleanCsv", "cleanDataFrame"]

def cleanDataFrame(df: pd.DataFrame, textCol: str = "text") -> pd.DataFrame:
    out = df.copy()
    if textCol not in out.columns:
        raise ValueError(f"Column '{textCol}' not found.")
    out[textCol] = out[textCol].fillna("").astype(str).map(cleanText)
    return out

def loadAndCleanCsv(path: str, textCol: str = "text") -> pd.DataFrame:
    df = pd.read_csv(path)
    return cleanDataFrame(df, textCol=textCol)
