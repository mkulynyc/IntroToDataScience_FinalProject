# nlp_utils.py
import re
from typing import Tuple, Set, Optional, Iterable
import nltk
import pandas as pd

from nlp.utils import cleanText
from nlp.vader_model import applyVader
from nlp.spacy_model import applySpacy

def initNltk(dataDir: str = ".nltk_data"):
    """
    Ensures required NLTK resources are available in a local folder.
    Returns: (SentimentIntensityAnalyzer instance, stopwords set)
    """
    if dataDir and dataDir not in nltk.data.path:
        nltk.data.path.append(dataDir)

    # Quiet, idempotent downloads
    for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "vader_lexicon"]:
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg, quiet=True, download_dir=dataDir)

    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    sw = set(stopwords.words("english"))

    # Domain-specific additions you might not want as tokens
    sw.update({"film", "movie", "series", "season", "episode", "netflix"})
    return sia, sw

def preprocessText(text: Optional[str]) -> str:
    """Lowercase & strip accents/punctuation-lite."""
    if not isinstance(text, str):
        return ""
    s = text.lower()
    # keep letters/numbers and spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenizeText(text: str, stopwordsSet: Set[str], minLen: int = 3) -> Iterable[str]:
    """Simple whitespace tokenization + stopword/short filter."""
    if not text:
        return []
    toks = text.split()
    return [t for t in toks if t not in stopwordsSet and len(t) >= minLen]

def computeSentiment(text: str, sia) -> float:
    """Return VADER compound score in [-1, 1]."""
    if not text:
        return 0.0
    return float(sia.polarity_scores(text)["compound"])

def addNlpColumns(df, sia, stopwordsSet):
    """
    Adds:
      - description_clean
      - tokens
      - sentiment_compound
    Mutates a copy of df and returns it.
    """
    import pandas as pd
    out = df.copy()
    if "description" not in out.columns:
        out["description"] = ""

    out["description_clean"] = out["description"].fillna("").apply(preprocessText)
    out["tokens"] = out["description_clean"].apply(lambda s: list(tokenizeText(s, stopwordsSet)))
    out["sentiment_compound"] = out["description"].fillna("").apply(lambda s: computeSentiment(s, sia))
    return out

def scoreDataFrame(
    df: pd.DataFrame,
    textCol: str = "text",
    spacyModelPath: str = "nlp/spacy_model/artifacts/best",
) -> pd.DataFrame:
    """
    Run the VADER + spaCy scoring pipeline on an arbitrary dataframe.
    Returns a copy with sentiment columns appended.
    """
    if textCol not in df.columns:
        raise ValueError(f"Missing text column: {textCol}")

    out = df.copy()
    out[textCol] = out[textCol].fillna("").astype(str).map(cleanText)
    out = applyVader(out, textCol=textCol)
    out = applySpacy(out, textCol=textCol, modelPath=spacyModelPath)
    return out
