import re
import time
import pathlib
import datetime as dt
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

def _cleanText(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()

def _safeGet(el, selector: str) -> str:
    found = el.select_one(selector)
    return _cleanText(found.get_text(" ", strip=True)) if found else ""

def _parseRating(text: str) -> Optional[float]:
    if not text:
        return None
    m1 = re.search(r"(\d+(?:\.\d+)?)/10", text)
    if m1:
        return float(m1.group(1))
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*out of\s*5", text, re.I)
    if m2:
        return float(m2.group(1)) * 2
    return None

def _parseDate(text: str) -> Optional[str]:
    text = _cleanText(text)
    for fmt in ("%B %d, %Y", "%d %B %Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    m = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return m.group(0) if m else None

def scrapeImdbReviews(url: str, maxPages: int = 1, sleepSec: float = 1.0) -> List[Dict]:
    reviews = []
    session = requests.Session()
    next_url = url
    for _ in range(maxPages):
        resp = session.get(next_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        page_title = _cleanText(soup.select_one("title").get_text() if soup.select_one("title") else "")
        inferred_title = page_title.replace("- User Reviews - IMDb", "").strip(" -")

        blocks = soup.select("div.review-container, div.lister-item-content")
        for b in blocks:
            author = _safeGet(b, "span.display-name-link a") or _safeGet(b, ".display-name-link a")
            review_text = _safeGet(b, "div.content div.text") or _safeGet(b, ".text.show-more__control")
            rating_text = _safeGet(b, "span.rating-other-user-rating") or _safeGet(b, ".ipl-ratings-bar")
            date_text = _safeGet(b, "span.review-date") or _safeGet(b, "span.review-date.ipl-inline-list__item")

            if review_text:
                reviews.append({
                    "title": inferred_title or "",
                    "source": "imdb",
                    "author": author,
                    "review_text": review_text,
                    "rating_10": _parseRating(rating_text),
                    "review_date": _parseDate(date_text),
                    "url": url
                })

        next_link = soup.select_one("a.lister-page-next.next-page")
        if not next_link or not next_link.get("href"):
            break
        next_url = "https://www.imdb.com" + next_link.get("href")
        time.sleep(sleepSec)
    return reviews

def scrapeRottenTomatoesAudience(url: str, maxPages: int = 1, sleepSec: float = 1.0) -> List[Dict]:
    reviews = []
    session = requests.Session()
    next_url = url
    for _ in range(maxPages):
        resp = session.get(next_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        title = _cleanText(soup.select_one("h1.scoreboard__title").get_text() if soup.select_one("h1.scoreboard__title") else "")
        blocks = soup.select("rt-review-card, div.review_table_row, div.audience-reviews__review-wrap")

        for b in blocks:
            txt = _cleanText(b.get_text(" ", strip=True))
            if not txt:
                continue
            rating = None
            m = re.search(r"(\d+(?:\.\d+)?)\/5", txt)
            if m:
                rating = float(m.group(1)) * 2
            date = None
            m2 = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}", txt)
            if m2:
                date = _parseDate(m2.group(0))
            reviews.append({
                "title": title,
                "source": "rottentomatoes",
                "author": "",
                "review_text": txt,
                "rating_10": rating,
                "review_date": date,
                "url": url
            })

        next_link = soup.select_one("a.js-prev-next-paging-next, a[data-qa='next-btn']")
        if not next_link or not next_link.get("href"):
            break
        href = next_link.get("href")
        next_url = href if href.startswith("http") else "https://www.rottentomatoes.com" + href
        time.sleep(sleepSec)
    return reviews

def scrapeReviews(url: str, site: str = "imdb", maxPages: int = 1) -> List[Dict]:
    site = site.lower().strip()
    if "imdb" in site:
        return scrapeImdbReviews(url, maxPages=maxPages)
    if "rotten" in site or site == "rt":
        return scrapeRottenTomatoesAudience(url, maxPages=maxPages)
    raise ValueError("Unsupported site. Try 'imdb' or 'rottentomatoes'.")

def saveReviewsToCsv(reviews: List[Dict], path: str) -> None:
    import pandas as pd
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(reviews)
    if p.exists():
        old = pd.read_csv(p)
        all_df = pd.concat([old, df], ignore_index=True)
        all_df.drop_duplicates(subset=["url", "author", "review_text"], inplace=True)
        all_df.to_csv(p, index=False)
    else:
        df.to_csv(p, index=False)
