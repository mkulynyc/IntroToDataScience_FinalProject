import re

__all__ = ["cleanText"]

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")

def cleanText(text: str) -> str:
    """
    Light, conservative cleaning for sentiment:
    - strip URLs
    - collapse whitespace
    - keep punctuation/emojis (helpful for VADER)
    """
    if text is None:
        return ""
    x = str(text)
    x = _URL_RE.sub("", x)
    x = _WS_RE.sub(" ", x).strip()
    return x
