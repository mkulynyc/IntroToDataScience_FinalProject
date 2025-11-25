import os
import time
from typing import List, Optional

import pandas as pd
import requests
import streamlit as st
from pathlib import Path
import sys
import plotly.express as px

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import custom modules
try:
    from nlp_utils import scoreDataFrame
except Exception:
    scoreDataFrame = None

try:
    from nlp.spacy_model import evaluateSpacy
except Exception:
    evaluateSpacy = None

try:
    from viz import plotVaderVsSpacy, plotLabelCounts
except Exception:
    plotVaderVsSpacy = None
    plotLabelCounts = None

# =========================
# Config & constants
# =========================
st.set_page_config(page_title="Netflix Sentiment Dashboard", layout="wide")

NETFLIX_PATH = "data/netflix_titles.csv"
REVIEWS_PATH = "data/reviews_raw.csv"
ENRICHED_PATH = "data/netflix_enriched_scored.csv"
SPACY_MODEL_PATH = "nlp/spacy_model/artifacts/best"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185"

# =========================
# Helpers
# =========================
@st.cache_data(show_spinner=False)
def loadCsv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        return None

@st.cache_data(show_spinner=False)
def loadEnriched(path: str) -> Optional[pd.DataFrame]:
    df = loadCsv(path)
    if df is None:
        return None
    # Ensure expected columns
    needed = {"title","type","release_year","nlp_text","vader_compound","vader_label","spacy_pos_prob","spacy_label"}
    missing = needed - set(df.columns)
    if missing:
        st.warning(f"Enriched file missing columns: {missing}")
    # derive genre list
    if "listed_in" in df.columns:
        df["genres_list"] = df["listed_in"].fillna("").astype(str).apply(lambda s: [g.strip() for g in s.split(",")] if s else [])
    else:
        df["genres_list"] = [[] for _ in range(len(df))]
    return df

def getUniqueSorted(values: pd.Series) -> List:
    return sorted([v for v in values.dropna().unique().tolist()])

def _getTmdbKey() -> Optional[str]:
    try:
        key = st.secrets.get("tmdb", {}).get("api_key")
        if key: return key
    except Exception:
        pass
    return os.getenv("TMDB_API_KEY")

@st.cache_data(show_spinner=False)
def fetchPosterPath(title: str, year: Optional[int], kind: str) -> Optional[str]:
    key = _getTmdbKey()
    if not key:
        return None
    try:
        kind = "movie" if str(kind).lower().startswith("movie") else "tv"
        params = {"api_key": key, "query": title}
        if year:
            params["year" if kind=="movie" else "first_air_date_year"] = int(year)
        r = requests.get(f"https://api.themoviedb.org/3/search/{kind}", params=params, timeout=12)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results: return None
        poster = results[0].get("poster_path")
        return poster
    except Exception:
        return None

# =========================
# Sidebar: Pipeline Controls
# =========================
with st.sidebar:
    st.header("Pipeline")
    st.caption("Fetch → Enrich/Score → Explore")
    fetch_btn = st.button("🔄 Fetch TMDB Reviews (append)", use_container_width=True)
    score_btn = st.button("⚙️ Enrich + Score (VADER + spaCy)", use_container_width=True)

    st.markdown("---")
    st.header("spaCy Evaluation")
    if st.button("Run spaCy Eval on test_reviews.csv", use_container_width=True):
        if evaluateSpacy is None:
            st.warning("spaCy evaluation module not available.")
        elif os.path.exists("data/test_reviews.csv"):
            try:
                m = evaluateSpacy("data/test_reviews.csv", textCol="text", labelCol="label", modelPath=SPACY_MODEL_PATH)
                st.success(f"Accuracy: {m['accuracy']:.3f} | Macro-F1: {m['macro_f1']:.3f}")
            except Exception as e:
                st.error(f"Eval failed: {e}")
        else:
            st.warning("data/test_reviews.csv not found.")

# =========================
# Dashboard Tabs
# =========================
tabs = st.tabs([
    "📊 Overview",
    "🔎 Explore",
    "⚖️ Model Compare",
    "🧠 Sentiment Analysis",
    "🧭 Title Explorer",
    "⚙️ Ingest & Score"
])

# ---------- 1. Overview ----------
with tabs[0]:
    df = loadEnriched(ENRICHED_PATH)
    st.header("Dashboard Overview")
    if df is None or df.empty:
        st.info("No enriched dataset yet.")
    else:
        total_titles = len(df)
        avg_vader = df["vader_compound"].mean() if "vader_compound" in df else 0
        pos_rate = (df["spacy_label"]=="POSITIVE").mean() if "spacy_label" in df else 0
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Titles", f"{total_titles}")
        c2.metric("Avg VADER Score", f"{avg_vader:.3f}")
        c3.metric("% Positive (spaCy)", f"{pos_rate*100:.1f}%")

# ---------- 2. Explore ----------
with tabs[1]:
    df = loadEnriched(ENRICHED_PATH)
    st.header("Explore Titles")
    if df is None or df.empty:
        st.info("No enriched dataset yet.")
    else:
        year_opt = ["(all)"] + getUniqueSorted(df["release_year"])
        kind_opt = ["(all)"] + getUniqueSorted(df["type"])
        year = st.selectbox("Filter by year", year_opt)
        kind = st.selectbox("Filter by type", kind_opt)
        mask = pd.Series(True, index=df.index)
        if year != "(all)":
            mask &= (df["release_year"]==year)
        if kind != "(all)":
            mask &= (df["type"]==kind)
        st.dataframe(df[mask][["title","type","release_year","vader_compound","spacy_label"]].head(100))

# ---------- 3. Model Compare ----------
with tabs[2]:
    df = loadEnriched(ENRICHED_PATH)
    st.header("VADER vs spaCy Comparison")
    if df is None or df.empty:
        st.info("No enriched dataset yet.")
    else:
        pos_thresh = st.slider("VADER Positive Threshold", 0.0, 0.5, 0.05)
        neg_thresh = st.slider("VADER Negative Threshold", -0.5, 0.0, -0.05)
        vlab = pd.Series("NEUTRAL", index=df.index)
        vlab = vlab.mask(df["vader_compound"]>=pos_thresh,"POSITIVE")
        vlab = vlab.mask(df["vader_compound"]<=neg_thresh,"NEGATIVE")
        agree = (vlab==df["spacy_label"])
        st.write(f"Agreement rate: {agree.mean()*100:.1f}%")
        try:
            fig = plotVaderVsSpacy(df)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.write("Scatter plot unavailable.")

# ---------- 4. Sentiment Analysis ----------
with tabs[3]:
    df = loadEnriched(ENRICHED_PATH)
    st.header("🧠 Sentiment Analysis")
    if df is None or df.empty:
        st.info("No enriched dataset yet.")
    else:
        # Top Positive / Negative Titles
        top_n = st.slider("Top N titles", 1, 20, 10)
        s = df.copy()
        s["spacy_pos_prob"] = s.get("spacy_pos_prob", pd.Series([0.0]*len(s))).fillna(0.0)
        s["vader_compound"] = s.get("vader_compound", pd.Series([0.0]*len(s))).fillna(0.0)
        s["vader_norm"] = (s["vader_compound"] + 1.0)/2.0
        s["combined_score"] = s["spacy_pos_prob"] + s["vader_norm"]
        pos = s.sort_values("combined_score", ascending=False).head(top_n)
        neg = s.sort_values("combined_score", ascending=True).head(top_n)

        left,right = st.columns(2)
        with left:
            st.subheader("Top Positive Titles")
            st.dataframe(pos[["title","type","release_year","combined_score"]])
        with right:
            st.subheader("Top Negative Titles")
            st.dataframe(neg[["title","type","release_year","combined_score"]])

        # Sentiment by Genre
        st.markdown("---")
        top_genres = st.slider("Top N genres", 1, 10, 5)
        genre_scores = s.explode("genres_list").groupby("genres_list").agg(
            avg_vader=("vader_compound","mean"),
            avg_spacy=("spacy_pos_prob","mean"),
            count=("title","count")
        ).sort_values("count", ascending=False).head(top_genres).reset_index()
        figg = px.bar(genre_scores, x="genres_list", y=["avg_vader","avg_spacy"], barmode="group", title="Avg Sentiment by Genre")
        st.plotly_chart(figg, use_container_width=True)

# ---------- 5. Title Explorer ----------
with tabs[4]:
    df = loadEnriched(ENRICHED_PATH)
    st.header("Title Explorer")
    if df is None or df.empty:
        st.info("No enriched dataset yet.")
    else:
        query = st.text_input("Search title")
        if query:
            sub = df[df["title"].str.contains(query, case=False, na=False)].head(25)
        else:
            sub = df.head(25)
        for _, row in sub.iterrows():
            st.subheader(row["title"])
            st.write(f"Type: {row.get('type','?')}, Year: {row.get('release_year','?')}")
            st.write(f"spaCy POS prob: {row.get('spacy_pos_prob',0.0):.3f}, VADER: {row.get('vader_compound',0.0):.3f}")

# ---------- 6. Ingest & Score ----------
with tabs[5]:
    st.header("Data Ingestion & Scoring")
    st.write("Use this tab to fetch reviews and score data")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Fetch TMDB Reviews"):
            st.write("Run fetch script here")
    with c2:
        if st.button("Enrich & Score"):
            st.write("Run scoring script here")
    st.markdown("---")
    st.file_uploader("Upload CSV to score (text column)", type=["csv"])
