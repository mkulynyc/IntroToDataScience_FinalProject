import os
import time
from typing import List, Optional

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import textstat

from nlp_utils import scoreDataFrame
from nlp.spacy_model import evaluateSpacy
from viz import plotVaderVsSpacy, plotLabelCounts

from plots import *
from engine import *
from data_load import *

# =========================
# Config & constants
# =========================
st.set_page_config(page_title="Netflix Sentiment (VADER + spaCy)", layout="wide")

NETFLIX_PATH = "data/netflix_titles.csv"
REVIEWS_PATH = "data/reviews_raw.csv"
ENRICHED_PATH = "data/netflix_enriched_scored.csv"
SPACY_MODEL_PATH = "nlp/spacy_model/artifacts/best_quick"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185"

# =========================
# Helpers (cached)
# =========================
@st.cache_data(show_spinner=False)
def loadCsv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    if os.path.getsize(path) == 0:
        st.warning(f"{path} is empty; generate it first.")
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
    # ensure expected columns exist
    needed = {"title","type","release_year","nlp_text",
              "vader_compound","vader_label","spacy_pos_prob","spacy_label"}
    missing = needed - set(df.columns)
    if missing:
        st.warning(f"Enriched file missing columns: {missing}")
    # derive genre list if present
    if "listed_in" in df.columns:
        df["genres_list"] = df["listed_in"].fillna("").astype(str).apply(
            lambda s: [g.strip() for g in s.split(",")] if s else []
        )
        # Extract primary genre (first genre in the list)
        df["primary_genre"] = df["genres_list"].apply(lambda lst: lst[0] if lst else 'Unknown')
    else:
        df["genres_list"] = [[] for _ in range(len(df))]
        df["primary_genre"] = "Unknown"
    return df

@st.cache_data(show_spinner=False)
def getUniqueSorted(values: pd.Series) -> List:
    return sorted([v for v in values.dropna().unique().tolist()])

def _getTmdbKey() -> Optional[str]:
    # Prefer Streamlit secrets, else env var
    try:
        key = st.secrets.get("tmdb", {}).get("api_key")
        if key: return key
    except Exception:
        pass
    return os.getenv("TMDB_API_KEY")

@st.cache_data(show_spinner=False)
def fetchPosterPath(title: str, year: Optional[int], kind: str) -> Optional[str]:
    """
    Best-effort TMDB poster fetcher (movie/tv).
    Safe to fail silently—UI will just skip posters.
    """
    key = _getTmdbKey()
    if not key:
        return None
    try:
        kind = "movie" if str(kind).lower().startswith("movie") else "tv"
        params = {"api_key": key, "query": title}
        if year:
            params["year" if kind == "movie" else "first_air_date_year"] = int(year)
        r = requests.get(f"https://api.themoviedb.org/3/search/{kind}", params=params, timeout=12)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        poster = results[0].get("poster_path")
        return poster
    except Exception:
        return None

def kpiCard(label: str, value: str, help_text: Optional[str] = None):
    st.metric(label, value, help=help_text)

def badge(text: str, tone: str = "neutral"):
    colors = {
        "positive": "#16a34a",  # green
        "negative": "#dc2626",  # red
        "neutral":  "#6b7280",  # gray
        "info":     "#2563eb"
    }
    color = colors.get(tone, "#6b7280")
    st.markdown(
        f"<span style='background:{color}22;color:{color};padding:.2rem .5rem;border-radius:999px;font-size:0.85rem'>{text}</span>",
        unsafe_allow_html=True
    )
# -------------------------------------------------------
# Description Analyzer Tab Function
# -------------------------------------------------------
def analyze_emotion(text):
    """Analyze emotion using TextBlob"""
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "sentiment": sentiment
        }
    except Exception as e:
        return {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "sentiment": "neutral"
        }

def analyze_readability(text):
    """Analyze readability using textstat"""
    try:
        score = textstat.flesch_reading_ease(str(text))
        
        if score >= 90:
            category = "Very Easy"
        elif score >= 80:
            category = "Easy"
        elif score >= 70:
            category = "Fairly Easy"
        elif score >= 60:
            category = "Standard"
        elif score >= 50:
            category = "Fairly Difficult"
        elif score >= 30:
            category = "Difficult"
        else:
            category = "Very Difficult"
        
        return {
            "score": score,
            "category": category
        }
    except Exception as e:
        return {
            "score": 0.0,
            "category": "unknown"
        }

def run_description_analysis(df):
    df = df.copy()

    df["description_emotion_polarity"] = 0.0
    df["description_emotion_subjectivity"] = 0.0
    df["description_emotion_sentiment"] = "neutral"
    df["description_readability_score"] = 0.0
    df["description_readability_category"] = "unknown"

    for i, row in df.iterrows():
        desc = row.get("description")
        if pd.isna(desc):
            continue

        emo = analyze_emotion(desc)
        df.loc[i, "description_emotion_polarity"] = emo["polarity"]
        df.loc[i, "description_emotion_subjectivity"] = emo["subjectivity"]
        df.loc[i, "description_emotion_sentiment"] = emo["sentiment"]

        read = analyze_readability(desc)
        df.loc[i, "description_readability_score"] = read["score"]
        df.loc[i, "description_readability_category"] = read["category"]

    return df


def description_analyzer_tab():
    st.header("📝 Description Analyzer")

    st.write("Upload a CSV with a **description** column to analyze sentiment & readability.")

    uploaded = st.file_uploader("Upload CSV", type=['csv'], key="description_analyzer_csv_uploader")

    if uploaded:
        df = pd.read_csv(uploaded)

        if "description" not in df.columns:
            st.error("Your CSV must contain a 'description' column.")
            return

        st.success(f"Loaded {len(df)} rows!")

        if st.button("Run Analysis", type="primary"):
            analyzed = run_description_analysis(df)

            st.success("Analysis complete!")
            st.subheader("Preview")
            st.dataframe(analyzed.head())

            # Sentiment chart
            st.subheader("Sentiment Polarity Distribution")
            fig1 = px.histogram(analyzed, x="description_emotion_polarity")
            fig1.update_traces(marker=dict(line=dict(color='white', width=2)))
            st.plotly_chart(fig1, use_container_width=True, key="desc_sentiment_chart")

            # Readability chart
            st.subheader("Readability Score Distribution")
            fig2 = px.histogram(analyzed, x="description_readability_score")
            fig2.update_traces(marker=dict(line=dict(color='white', width=2)))
            st.plotly_chart(fig2, use_container_width=True, key="desc_readability_chart")

            csv = analyzed.to_csv(index=False)
            st.download_button("Download Results", csv, "description_analysis.csv")

    st.markdown("---")

    # Single text analyzer
    st.subheader("Analyze a Single Description")
    text = st.text_area("Enter description:")

    if text:
        emo = analyze_emotion(text)
        read = analyze_readability(text)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", emo["sentiment"])
            st.metric("Polarity", emo["polarity"])
            st.metric("Subjectivity", emo["subjectivity"])

        with col2:
            st.metric("Reading Level", read["category"])
            st.metric("Score", read["score"])
# =========================
# Sidebar (pipeline actions)
# =========================
with st.sidebar:
    st.header("Pipeline")
    st.caption("Fetch → Enrich/Score → Explore")
    fetch_btn = st.button("🔄 Fetch TMDB Reviews (append)", use_container_width=True)
    score_btn = st.button("⚙️ Enrich + Score (VADER + spaCy)", use_container_width=True)

    st.markdown("---")
    st.header("spaCy Evaluation")
    if st.button("Run Eval on data/test_reviews.csv", use_container_width=True):
        if os.path.exists("data/test_reviews.csv"):
            try:
                m = evaluateSpacy("data/test_reviews.csv", textCol="text", labelCol="label",
                                  modelPath=SPACY_MODEL_PATH)
                st.success(f"Accuracy: {m['accuracy']:.3f} | Macro-F1: {m['macro_f1']:.3f}")
                with st.expander("Confusion matrix"):
                    st.write("Labels:", m["confusion_matrix_labels"])
                    st.write(m["confusion_matrix"])
            except Exception as e:
                st.error(f"Eval failed: {e}")
        else:
            st.warning("data/test_reviews.csv not found.")

    st.markdown("---")
    st.caption("Tip: set your TMDB API key in `.streamlit/secrets.toml` or TMDB_API_KEY env.")

# Run CLI scripts with feedback
def runScript(cmd: str):
    with st.status(f"Running: `{cmd}`", expanded=True) as status:
        start = time.time()
        st.write("Starting…")
        code = os.system(cmd)
        if code != 0:
            status.update(label=f"❌ Failed (exit {code})", state="error")
            st.error(f"Command failed (exit {code})")
        else:
            secs = time.time() - start
            status.update(label=f"✅ Done in {secs:.1f}s", state="complete")
            st.success("Completed.")

if fetch_btn:
    if not os.path.exists(NETFLIX_PATH):
        st.error("Missing data/netflix_titles.csv.")
    else:
        runScript(f'python "scripts/fetch_tmdb_reviews.py" --netflix "{NETFLIX_PATH}" --output "{REVIEWS_PATH}" --limit 300')

if score_btn:
    if not os.path.exists(NETFLIX_PATH):
        st.error("Missing data/netflix_titles.csv.")
    elif not os.path.exists(REVIEWS_PATH):
        st.error("Missing data/reviews_raw.csv — click “Fetch TMDB Reviews” first.")
    else:
        runScript(f'python "scripts/enrich_and_score.py" --netflix "{NETFLIX_PATH}" --reviews "{REVIEWS_PATH}" --output "{ENRICHED_PATH}" --spacyModel "{SPACY_MODEL_PATH}"')

# =========================
# Tabs
# =========================
st.title("🎬 Netflix Sentiment Workbench (VADER + spaCy)")

tabs = st.tabs([
    "📊 Overview",
    "🔎 Explore",
    "⚖️ Model Compare",
    "🧭 Title Explorer",
    "⚙️ Ingest & Score",
    "🎯 Recommender Engine",
    "📊 Visualizations",
    "📈 Statistics DK", 
    "🔍 Analysis DK", 
    "⏰ Time Series DK",
    "🎨 Visualizations DK",
    "📝 Description Analyzer"
])

# ---------- Overview ----------
with tabs[0]:
    df = loadEnriched(ENRICHED_PATH)
    if df is None or df.empty:
        st.info("No enriched dataset yet. Use the **Ingest & Score** tab (or sidebar buttons) to create it.")
    else:
        # KPIs
        total_titles = len(df)
        pos_rate = (df["spacy_label"] == "POSITIVE").mean() if "spacy_label" in df else 0.0
        avg_vader = df["vader_compound"].mean() if "vader_compound" in df else 0.0
        most_reviewed = None
        if "review_join" in df.columns:
            review_counts = df["review_join"].fillna("").astype(str).apply(lambda s: 0 if not s else s.count(" || ") + 1)
            if len(review_counts):
                idx = review_counts.idxmax()
                most_reviewed = df.loc[idx, "title"]

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpiCard("Titles scored", f"{total_titles:,}")
        with c2: kpiCard("spaCy % Positive", f"{pos_rate*100:,.1f}%")
        with c3: kpiCard("Avg VADER compound", f"{avg_vader:,.3f}")
        with c4: kpiCard("Most reviewed", most_reviewed or "—")

        st.markdown("### Highlights")
        left, right = st.columns(2)
        with left:
            try:
                fig = plotLabelCounts(df, which="spacy_label")
                st.plotly_chart(fig, use_container_width=True, key="overview_label_counts")
            except Exception as e:
                st.warning(f"Chart error: {e}")
        with right:
            try:
                scatter = plotVaderVsSpacy(df, textCol="nlp_text")
                st.plotly_chart(scatter, use_container_width=True, key="overview_vader_spacy")
            except Exception as e:
                st.warning(f"Chart error: {e}")

# ---------- Explore ----------
with tabs[1]:
    df = loadEnriched(ENRICHED_PATH)
    if df is None or df.empty:
        st.info("Create an enriched dataset first.")
    else:
        st.subheader("Filters")
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        with c1:
            year_opt = ["(all)"] + getUniqueSorted(df["release_year"])
            year = st.selectbox("Release year", year_opt)
        with c2:
            types_opt = ["(all)"] + getUniqueSorted(df["type"])
            kind = st.selectbox("Type", types_opt)
        with c3:
            spacy_label = st.selectbox("spaCy label", ["(all)","POSITIVE","NEGATIVE"])
        with c4:
            min_conf = st.slider("Min spaCy POS prob", 0.0, 1.0, 0.0, 0.01)

        # Genres (multi-select) if present
        genres_flat = sorted({g for lst in df["genres_list"] for g in lst}) if "genres_list" in df.columns else []
        selected_genres = st.multiselect("Genres (any)", genres_flat)

        mask = pd.Series(True, index=df.index)
        if year != "(all)": mask &= (df["release_year"] == year)
        if kind != "(all)": mask &= (df["type"] == kind)
        if spacy_label != "(all)": mask &= (df["spacy_label"] == spacy_label)
        if min_conf > 0:
            mask &= (df["spacy_pos_prob"] >= min_conf)
        if selected_genres:
            mask &= df["genres_list"].apply(lambda lst: any(g in lst for g in selected_genres))

        view = df[mask].copy()
        st.caption(f"Showing {len(view):,} of {len(df):,} rows")

        # Pretty table
        show_cols = ["title","type","release_year","spacy_label","spacy_pos_prob","vader_label","vader_compound"]
        if "listed_in" in view.columns:
            show_cols.append("listed_in")
        st.dataframe(view[show_cols].head(400))

        # Charts
        st.markdown("### Charts")
        colA, colB = st.columns(2)
        with colA:
            try:
                fig1 = plotVaderVsSpacy(view, textCol="nlp_text")
                st.plotly_chart(fig1, use_container_width=True, key="explore_vader_spacy")
            except Exception as e:
                st.warning(f"Chart error: {e}")
        with colB:
            try:
                fig2 = plotLabelCounts(view, which="spacy_label")
                st.plotly_chart(fig2, use_container_width=True, key="explore_label_counts")
            except Exception as e:
                st.warning(f"Chart error: {e}")

        st.download_button("Download current view as CSV",
                           data=view.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_view.csv",
                           use_container_width=True)

# ---------- Model Compare (agreement / disagreement) ----------
with tabs[2]:
    df = loadEnriched(ENRICHED_PATH)
    if df is None or df.empty:
        st.info("Create an enriched dataset first.")
    else:
        st.subheader("Agreement & Thresholds")
        c1, c2 = st.columns(2)
        with c1:
            pos_thresh = st.slider("VADER positive threshold", 0.0, 0.5, 0.05, 0.01)
        with c2:
            neg_thresh = st.slider("VADER negative threshold", -0.5, 0.0, -0.05, 0.01)

        # recompute VADER label on the fly for comparison, if desired
        vlab = pd.Series("NEUTRAL", index=df.index)
        vlab = vlab.mask(df["vader_compound"] >= pos_thresh, "POSITIVE")
        vlab = vlab.mask(df["vader_compound"] <= neg_thresh, "NEGATIVE")

        agree = (vlab == df["spacy_label"])
        st.write(f"**Agreement rate:** {(agree.mean()*100):.1f}%  (n={len(df)})")

        # Show disagreements table
        st.markdown("#### Disagreements")
        dis = df[~agree].copy()
        st.dataframe(dis[["title","type","release_year","spacy_label","spacy_pos_prob","vader_compound","vader_label"]].head(300))

        st.markdown("#### VADER vs spaCy")
        try:
            scatter = plotVaderVsSpacy(df, textCol="nlp_text")
            st.plotly_chart(scatter, use_container_width=True, key="model_compare_vader_spacy")
        except Exception as e:
            st.warning(f"Chart error: {e}")

# ---------- Title Explorer ----------
with tabs[3]:
    df = loadEnriched(ENRICHED_PATH)
    if df is None or df.empty:
        st.info("Create an enriched dataset first.")
    else:
        st.subheader("Find a title")
        query = st.text_input("Search title", placeholder="Start typing…")
        if query:
            sub = df[df["title"].str.contains(query, case=False, na=False)].head(25).copy()
        else:
            sub = df.head(25).copy()

        for _, row in sub.iterrows():
            with st.container(border=True):
                top = st.columns([1, 3, 2])
                # Poster
                with top[0]:
                    poster = fetchPosterPath(row["title"], row.get("release_year"), row.get("type","movie"))
                    if poster:
                        st.image(f"{TMDB_IMAGE_BASE}{poster}")
                    else:
                        st.write("No image")
                # Meta
                with top[1]:
                    st.subheader(str(row["title"]))
                    meta = f"{row.get('type','?')} • {row.get('release_year','?')}"
                    st.caption(meta)
                    # badges
                    badge(f"spaCy: {row.get('spacy_label','?')}",
                          "positive" if row.get("spacy_label")=="POSITIVE" else "negative")
                    st.write(f"spaCy pos prob: {row.get('spacy_pos_prob',0.0):.3f}")
                    badge(f"VADER: {row.get('vader_label','?')}",
                          "positive" if row.get("vader_label")=="POSITIVE" else ("negative" if row.get("vader_label")=="NEGATIVE" else "neutral"))
                    st.write(f"VADER compound: {row.get('vader_compound',0.0):.3f}")
                # Text
                with top[2]:
                    txt = str(row.get("nlp_text","")).strip()
                    if len(txt) > 280:
                        st.text_area("Text (truncated)", value=txt[:800] + ("…" if len(txt)>800 else ""), height=160)
                        with st.expander("Show full text"):
                            st.write(txt)
                    else:
                        st.text_area("Text", value=txt, height=160)

# ---------- Ingest & Score ----------
with tabs[4]:
    st.subheader("Ingest & Score")
    st.write("Run the end-to-end pipeline from within the UI, or use the sidebar buttons.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Fetch TMDB Reviews now", use_container_width=True):
            if not os.path.exists(NETFLIX_PATH):
                st.error("Missing data/netflix_titles.csv.")
            else:
                runScript(f'python "scripts/fetch_tmdb_reviews.py" --netflix "{NETFLIX_PATH}" --output "{REVIEWS_PATH}" --limit 300')
    with c2:
        if st.button("⚙️ Enrich + Score now", use_container_width=True):
            if not os.path.exists(NETFLIX_PATH):
                st.error("Missing data/netflix_titles.csv.")
            elif not os.path.exists(REVIEWS_PATH):
                st.error("Missing data/reviews_raw.csv — fetch reviews first.")
            else:
                runScript(f'python "scripts/enrich_and_score.py" --netflix "{NETFLIX_PATH}" --reviews "{REVIEWS_PATH}" --output "{ENRICHED_PATH}" --spacyModel "{SPACY_MODEL_PATH}"')

    st.markdown("---")
    st.caption("Need to just score a small CSV on the fly? Upload it below (uses your trained spaCy model).")
    up = st.file_uploader("Upload CSV with a 'text' column", type=["csv"])
    if up:
        try:
            raw = pd.read_csv(up)
            scored = scoreDataFrame(raw, textCol="text", spacyModelPath=SPACY_MODEL_PATH)
            st.success("Scored! Preview below.")
            st.dataframe(scored.head(100))
            st.download_button("Download scored CSV",
                               data=scored.to_csv(index=False).encode("utf-8"),
                               file_name="scored.csv",
                               use_container_width=True)
        except Exception as e:
            st.error(f"Scoring failed: {e}")
 
# ---------- Recommender Search Engine ---------- 
with tabs[5]:
    # Load data
    df_raw = loadCsv(NETFLIX_PATH)
    df_clean, _ = cleanNetflixData(df_raw)
    df_clean = add_genres_list(df_clean)
    
    # Set up recommender engine on streamlit
    st.subheader("Recommender Search Engine")
    st.write("Write key words and select genres you are interested in, and choose if you want all or any of these in the search.")
    
    # Inline filters
    col1, col2 = st.columns(2)
    with col1:
        keywords_input = st.text_input("Keywords (comma-separated)", value="school")
        keyword_mode = st.radio("Keyword Match Mode", ["any", "all"])
    with col2:
        selected_genres = st.multiselect(
            "Genres",
            options=sorted(set(g for sublist in df_clean['genres_list'] for g in sublist if g))
        )
        genre_mode = st.radio("Genre Match Mode", ["any", "all"])

    top_n = st.slider("Number of Recommendations", 1, 20, 10)

    # Run recommender
    st.subheader("Recommended Titles")
    if st.button("Run Recommender"):
        keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
        results = run_recommender(df_clean, keywords, selected_genres, top_n, keyword_mode, genre_mode)
        if results.empty:
            st.warning("No matches found. Try different keywords or genres.")
        else:
            st.dataframe(results.reset_index(drop=True), use_container_width=True)
            
# ---------- Visualizations ---------
with tabs[6]:
    # Load data
    df_raw = loadCsv(NETFLIX_PATH)
    df_clean, _ = cleanNetflixData(df_raw)
    df_clean = add_genres_list(df_clean)
    
    
    st.header("📊 Netflix Content Visualizations")

    # Ratings table with year slider
    st.subheader("🎬 Ratings Table")
    year_cutoff = st.slider("Minimum Release Year", min_value=1980, max_value=2025, value=2016)
    show_rating_table(df_clean, year=year_cutoff)

    # Top genres by country
    st.subheader("🌍 Top Genres by Country Over Time")
    country = st.text_input("Enter a country", value="United States")
    top_n_genres = st.slider("Top N Genres", 3, 10, 5)
    plot_top_genres_by_country(df_clean, country=country, top_n=top_n_genres)

# ---------- Statistics DK ----------
with tabs[7]:
    df = loadEnriched(ENRICHED_PATH)
    if df is None or df.empty:
        st.info("No enriched dataset yet. Please create the enriched dataset first.")
    else:
        st.subheader("📈 Statistical Analysis DK")
        
        # Try to load and merge runtime data if available
        runtime_path = "netflix_movies_tv_runtime.csv"
        if os.path.exists(runtime_path):
            try:
                runtime_df = pd.read_csv(runtime_path)
                # Merge on title (or show_id if available)
                if 'title' in df.columns and 'title' in runtime_df.columns:
                    df = df.merge(runtime_df[['title', 'rating_stars']], on='title', how='left', suffixes=('', '_runtime'))
                    st.success(f"✅ Merged runtime data from {runtime_path}")
            except Exception as e:
                st.warning(f"Could not load runtime data: {e}")
        
        # Descriptive statistics - Duration
        st.subheader("🎬 Content Duration Statistics")
        if 'release_year' in df.columns:
            release_year = pd.to_numeric(df['release_year'], errors='coerce').dropna()
            if not release_year.empty:
                year_stats = release_year.describe()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Release Year Statistics**")
                    st.dataframe(year_stats.round(0))
                
                with col2:
                    fig_year_hist = px.histogram(
                        release_year,
                        nbins=20,
                        title="Release Year Distribution",
                        labels={'value': 'Release Year', 'count': 'Number of Titles'}
                    )
                    fig_year_hist.update_traces(marker=dict(line=dict(color='white', width=1)))
                    st.plotly_chart(fig_year_hist, use_container_width=True, key="stats_year_hist")
            else:
                st.info("Release year data not available after processing.")
        
        # Rating distribution
        # Check for TMDB rating from runtime CSV (prioritize _runtime suffix to get numeric TMDB ratings)
        st.subheader("⭐ Rating Analysis")
        rating_col = None
        
        # First check all columns to find any numeric rating column
        for col in df.columns:
            if 'rating' in col.lower() and ('_runtime' in col.lower() or 'stars' in col.lower()):
                # Try to check if it's numeric
                try:
                    test_vals = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(test_vals) > 0 and test_vals.max() <= 10:  # TMDB ratings are 0-10
                        rating_col = col
                        break
                except:
                    continue
        
        # Fallback to checking specific column names
        if not rating_col:
            possible_rating_cols = ['rating_stars', 'rating_stars_runtime', 'rating_runtime', 'total_rating_runtime', 'total_rating', 'vote_average', 'tmdb_rating']
            for col in possible_rating_cols:
                if col in df.columns:
                    rating_col = col
                    break
            
        if rating_col:
            try:
                st.write(f"**Using column: `{rating_col}`**")
                
                # Get ratings, convert to numeric, and clean data
                ratings = pd.to_numeric(df[rating_col], errors='coerce').dropna()
                
                if len(ratings) > 0:
                    # Create histogram with rating bins
                    fig_rating = px.histogram(
                        x=ratings,
                        nbins=20,
                        title="TMDB Rating Distribution (0-10 scale)",
                        labels={'x': 'Rating (0-10)', 'y': 'Number of Titles'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_rating.update_traces(marker=dict(line=dict(color='white', width=1)))
                    fig_rating.update_layout(
                        xaxis_title="Rating (0-10)",
                        yaxis_title="Number of Titles",
                        showlegend=False
                    )
                    st.plotly_chart(fig_rating, use_container_width=True, key="stats_rating_hist")
                    
                    # Show rating statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Rating", f"{ratings.mean():.2f}/10")
                    with col2:
                        st.metric("Median Rating", f"{ratings.median():.2f}/10")
                    with col3:
                        st.metric("Total Rated", f"{len(ratings):,}")
                else:
                    st.info(f"No valid rating data found in column '{rating_col}'.")
            except Exception as e:
                st.error(f"Error displaying rating distribution: {str(e)}")
        else:
            # Debug: Show what columns are available
            with st.expander("🔍 Debug: Available columns"):
                st.write("Looking for TMDB rating columns. All columns with 'rating':")
                rating_cols = [col for col in df.columns if 'rating' in col.lower()]
                st.write(rating_cols)
                st.write("\nAll columns:")
                st.write(list(df.columns))
            st.info("No TMDB rating column found. Please ensure netflix_movies_tv_runtime.csv is loaded correctly.")

# ---------- Analysis DK ----------
with tabs[8]:
    df = loadEnriched(ENRICHED_PATH)
    if df is None or df.empty:
        st.info("No enriched dataset yet. Please create the enriched dataset first.")
    else:
        st.subheader("🔍 Advanced Analysis DK")
        
        # Show data summary
        st.write("**Dataset Summary**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Titles", f"{len(df):,}")
        with col2:
            if "spacy_label" in df.columns:
                pos_count = (df["spacy_label"] == "POSITIVE").sum()
                st.metric("Positive Sentiment", f"{pos_count:,}")
        with col3:
            if "vader_compound" in df.columns:
                avg_compound = df["vader_compound"].mean()
                st.metric("Avg VADER Score", f"{avg_compound:.3f}")
        
        # Correlation analysis
        st.subheader("📊 Sentiment Correlation")
        if "vader_compound" in df.columns and "spacy_pos_prob" in df.columns:
            correlation = df[["vader_compound", "spacy_pos_prob"]].corr().iloc[0, 1]
            st.write(f"**Correlation between VADER and spaCy scores:** {correlation:.3f}")
            
            # Scatter plot
            fig_corr = px.scatter(
                df.sample(min(1000, len(df))),
                x="vader_compound",
                y="spacy_pos_prob",
                title="VADER vs spaCy Sentiment Scores",
                labels={'vader_compound': 'VADER Compound Score', 'spacy_pos_prob': 'spaCy Positive Probability'}
            )
            st.plotly_chart(fig_corr, use_container_width=True, key="analysis_correlation")
        
        # Clustering analysis
        st.subheader("🎯 Content Clustering")
        
        # Check if we have the necessary columns for clustering
        if "vader_compound" in df.columns and "spacy_pos_prob" in df.columns and "release_year" in df.columns:
            # Create clustering based on sentiment features
            st.write("**Performing K-Means clustering based on sentiment and temporal features...**")
            
            # Prepare features for clustering
            cluster_df = df.copy()
            cluster_df['release_year_numeric'] = pd.to_numeric(cluster_df['release_year'], errors='coerce')
            
            # Select features and remove NaN
            features_df = cluster_df[['vader_compound', 'spacy_pos_prob', 'release_year_numeric']].dropna()
            
            if len(features_df) > 10:  # Need enough data points
                # Number of clusters
                n_clusters = st.slider("Number of clusters:", 2, 8, 4, key="analysis_n_clusters")
                
                # Standardize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_df)
                
                # Perform K-Means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features_scaled)
                
                # Add cluster labels back to the dataframe
                features_df['cluster'] = cluster_labels
                cluster_df.loc[features_df.index, 'cluster_label'] = cluster_labels
                
                # Visualize clusters
                col1, col2 = st.columns(2)
                
                with col1:
                    # Cluster distribution
                    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
                    fig_clusters = px.bar(
                        x=cluster_counts.index,
                        y=cluster_counts.values,
                        title="Content Clusters Distribution",
                        labels={'x': 'Cluster', 'y': 'Number of Titles'}
                    )
                    st.plotly_chart(fig_clusters, use_container_width=True, key="analysis_clusters")
                
                with col2:
                    # Cluster visualization in 2D (VADER vs spaCy)
                    fig_scatter = px.scatter(
                        features_df,
                        x='vader_compound',
                        y='spacy_pos_prob',
                        color=features_df['cluster'].astype(str),
                        title="Clusters in Sentiment Space",
                        labels={'vader_compound': 'VADER Compound', 'spacy_pos_prob': 'spaCy Positive Prob', 'color': 'Cluster'}
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True, key="analysis_cluster_scatter")
                
                # Show cluster statistics
                st.subheader("📊 Cluster Statistics")
                cluster_stats = features_df.groupby('cluster').agg({
                    'vader_compound': ['mean', 'std'],
                    'spacy_pos_prob': ['mean', 'std'],
                    'release_year_numeric': ['mean', 'min', 'max']
                }).round(3)
                st.dataframe(cluster_stats, use_container_width=True)
                
                # Show sample from each cluster
                st.subheader("📋 Sample Content by Cluster")
                selected_cluster = st.selectbox(
                    "Select cluster to view sample content:",
                    options=sorted(features_df['cluster'].unique()),
                    key="analysis_cluster_selector"
                )
                
                # Get sample from selected cluster
                cluster_indices = features_df[features_df['cluster'] == selected_cluster].index
                cluster_sample = df.loc[cluster_indices].head(10)
                
                # Build column list based on availability
                display_cols = ['title']
                if 'type' in cluster_sample.columns:
                    display_cols.append('type')
                if 'release_year' in cluster_sample.columns:
                    display_cols.append('release_year')
                if 'spacy_label' in cluster_sample.columns:
                    display_cols.append('spacy_label')
                if 'vader_compound' in cluster_sample.columns:
                    display_cols.append('vader_compound')
                
                st.dataframe(
                    cluster_sample[display_cols],
                    use_container_width=True
                )
            else:
                st.warning("Not enough data points for clustering (need at least 10 complete records).")
        else:
            st.info("Clustering requires vader_compound, spacy_pos_prob, and release_year columns in the dataset.")

# ---------- Time Series DK ----------
with tabs[9]:
    df = loadEnriched(ENRICHED_PATH)
    if df is None or df.empty:
        st.info("No enriched dataset yet. Please create the enriched dataset first.")
    else:
        st.subheader("⏰ Time Series Analysis DK")
        
        if 'release_year' in df.columns:
            release_years = pd.to_numeric(df['release_year'], errors='coerce').dropna().astype(int)
            if not release_years.empty:
                year_counts = release_years.value_counts().sort_index()
                
                fig_trend = px.line(
                    x=year_counts.index,
                    y=year_counts.values,
                    title="Netflix Content by Release Year",
                    labels={'x': 'Release Year', 'y': 'Number of Titles'}
                )
                fig_trend.update_traces(line=dict(color='#e50914', width=2))
                st.plotly_chart(fig_trend, use_container_width=True, key="timeseries_releases")
                
                # Sentiment over time
                if "spacy_label" in df.columns:
                    st.subheader("📈 Sentiment Trend Over Time")
                    df_time = df.copy()
                    df_time['release_year'] = pd.to_numeric(df_time['release_year'], errors='coerce')
                    df_time = df_time.dropna(subset=['release_year', 'spacy_label'])
                    
                    if not df_time.empty:
                        sentiment_by_year = (
                            df_time
                            .groupby(['release_year', 'spacy_label'])
                            .size()
                            .reset_index(name='count')
                        )
                        
                        fig_sentiment = px.line(
                            sentiment_by_year,
                            x='release_year',
                            y='count',
                            color='spacy_label',
                            title="Sentiment Distribution Over Years",
                            labels={'release_year': 'Release Year', 'count': 'Number of Titles'}
                        )
                        st.plotly_chart(fig_sentiment, use_container_width=True, key="timeseries_sentiment")
            else:
                st.info("Release year data not available for time series analysis.")
        else:
            st.warning("Release year column not found in dataset.")

# ---------- Visualizations DK ----------
with tabs[10]:
    df = loadEnriched(ENRICHED_PATH)
    if df is None or df.empty:
        st.info("No enriched dataset yet. Please create the enriched dataset first.")
    else:
        st.subheader("🎨 Interactive Visualizations DK")
        
        # Genre analysis
        st.subheader("🎭 Genre Analysis")
        
        if 'primary_genre' in df.columns:
            genre_counts = df['primary_genre'].value_counts().head(15)
            fig_genres = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation='h',
                title="Top Genres / Runtime Buckets"
            )
            st.plotly_chart(fig_genres, use_container_width=True, key="viz_primary_genres")
            
            if 'content_minutes' in df.columns:
                genre_runtime = (
                    df[['primary_genre', 'content_minutes']]
                    .dropna()
                    .groupby('primary_genre')['content_minutes']
                    .mean()
                    .sort_values(ascending=False)
                    .head(15)
                )
                if not genre_runtime.empty:
                    fig_runtime = px.bar(
                        x=genre_runtime.values,
                        y=genre_runtime.index,
                        orientation='h',
                        title="Average Content Minutes by Genre"
                    )
                    st.plotly_chart(fig_runtime, use_container_width=True, key="viz_genre_runtime")
        else:
            st.info("Genre metadata unavailable; showing runtime and region trends instead.")
        
        # Country distribution
        if 'country' in df.columns:
            all_countries = []
            for countries in df['country'].dropna():
                if isinstance(countries, list):
                    all_countries.extend([c for c in countries if c])
                elif isinstance(countries, str):
                    all_countries.extend([c.strip() for c in countries.split(',') if c])
            
            if all_countries:
                country_series = pd.Series(all_countries, dtype="object").astype(str).str.strip()
                exclusions = {'', 'unknown', 'nan', 'none', 'n/a'}
                country_series = country_series[~country_series.str.casefold().isin(exclusions)]
                if not country_series.empty:
                    country_counts = country_series.value_counts().head(15)
                    fig_countries = px.bar(
                        x=country_counts.values,
                        y=country_counts.index,
                        orientation='h',
                        title="Top 15 Countries by Content Count"
                    )
                    st.plotly_chart(fig_countries, use_container_width=True, key="viz_countries")
                else:
                    st.info("Country information not available for visualization.")
            else:
                st.info("Country information not available for visualization.")
        
        # Release year vs Rating
        if 'release_year' in df.columns and 'rating' in df.columns:
            trend_df = df.copy()
            trend_df['release_year'] = pd.to_numeric(trend_df['release_year'], errors='coerce')
            trend_df = trend_df.dropna(subset=['release_year', 'rating'])
            
            if not trend_df.empty:
                trend_df['release_year'] = trend_df['release_year'].astype(int)
                rating_counts = (
                    trend_df
                    .groupby(['release_year', 'rating'])
                    .size()
                    .reset_index(name='count')
                )
                rating_counts['rating'] = rating_counts['rating'].astype(str)
                rating_counts = rating_counts[
                    ~rating_counts['rating'].str.casefold().isin({'unknown', 'nan', 'none', ''})
                ]
                if not rating_counts.empty:
                    rating_counts = rating_counts.sort_values(['release_year', 'rating'])
                    fig_rating_trend = px.line(
                        rating_counts,
                        x='release_year',
                        y='count',
                        color='rating',
                        markers=True,
                        title="Content Rating Trend by Release Year",
                        labels={'count': 'Number of Titles', 'release_year': 'Release Year'}
                    )
                    st.plotly_chart(fig_rating_trend, use_container_width=True, key="viz_rating_trend")
                else:
                    st.info("Rating trend chart skipped: no rating data available.")
            else:
                st.info("Insufficient release year data for rating trends.")
        
        # Type distribution with sentiment
        if "type" in df.columns and "spacy_label" in df.columns:
            st.subheader("📺 Content Type vs Sentiment")
            
            type_sentiment = (
                df.groupby(['type', 'spacy_label'])
                .size()
                .reset_index(name='count')
            )
            
            fig_type_sentiment = px.bar(
                type_sentiment,
                x='type',
                y='count',
                color='spacy_label',
                title="Content Type by Sentiment",
                labels={'count': 'Number of Titles', 'type': 'Content Type'},
                barmode='group'
            )
            st.plotly_chart(fig_type_sentiment, use_container_width=True, key="viz_type_sentiment")
        
        # VADER score distribution
        if "vader_compound" in df.columns:
            st.subheader("📊 VADER Score Distribution")
            
            fig_vader = px.histogram(
                df,
                x="vader_compound",
                nbins=50,
                title="Distribution of VADER Compound Scores",
                labels={'vader_compound': 'VADER Compound Score', 'count': 'Number of Titles'}
            )
            fig_vader.update_traces(marker=dict(color='#564d4d', line=dict(color='white', width=1)))
            st.plotly_chart(fig_vader, use_container_width=True, key="viz_vader_dist")

# ---------- Description Analyzer ----------
with tabs[11]:
    description_analyzer_tab()