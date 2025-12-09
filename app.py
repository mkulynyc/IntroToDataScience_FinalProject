import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
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
    needed = {
        "title",
        "type",
        "release_year",
        "nlp_text",
        "vader_compound",
        "vader_label",
        "spacy_pos_prob",
        "spacy_label",
    }
    missing = needed - set(df.columns)
    if missing:
        st.warning(f"Enriched file missing columns: {missing}")
    # derive genre list if present
    if "listed_in" in df.columns:
        df["genres_list"] = df["listed_in"].fillna("").astype(str).apply(
            lambda s: [g.strip() for g in s.split(",")] if s else []
        )
        # Extract primary genre (first genre in the list)
        df["primary_genre"] = df["genres_list"].apply(
            lambda lst: lst[0] if lst else "Unknown"
        )
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
        if key:
            return key
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
        r = requests.get(
            f"https://api.themoviedb.org/3/search/{kind}",
            params=params,
            timeout=12,
        )
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
        "neutral": "#6b7280",  # gray
        "info": "#2563eb",
    }
    color = colors.get(tone, "#6b7280")
    st.markdown(
        f"<span style='background:{color}22;color:{color};padding:.2rem .5rem;"
        f"border-radius:999px;font-size:0.85rem'>{text}</span>",
        unsafe_allow_html=True,
    )

# -------------------------------------------------------
# Description Analyzer helpers (NEW VERSION)
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
            "sentiment": sentiment,
        }
    except Exception:
        return {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "sentiment": "neutral",
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
            "category": category,
        }
    except Exception:
        return {
            "score": 0.0,
            "category": "unknown",
        }


def run_description_analysis(df: pd.DataFrame) -> pd.DataFrame:
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

    st.write("Analyzing descriptions from **netflix_titles.csv**")

    # Load the default Netflix CSV
    df = None
    if os.path.exists(NETFLIX_PATH):
        try:
            df = pd.read_csv(NETFLIX_PATH)
            st.success(f"✅ Loaded {len(df)} rows from {NETFLIX_PATH}")
        except Exception as e:
            st.error(f"Error loading {NETFLIX_PATH}: {e}")
            return
    else:
        st.error(f"File not found: {NETFLIX_PATH}")
        return

    if "description" not in df.columns:
        st.error("The CSV must contain a 'description' column.")
        return

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Analyzing descriptions..."):
            analyzed = run_description_analysis(df)

        st.success("Analysis complete!")
        st.subheader("Preview")
        st.dataframe(analyzed.head())

        # Sentiment chart
        st.subheader("Sentiment Polarity Distribution")
        fig1 = px.histogram(analyzed, x="description_emotion_polarity")
        fig1.update_traces(marker=dict(line=dict(color="white", width=2)))
        st.plotly_chart(fig1, use_container_width=True, key="desc_sentiment_chart")

        # Readability chart
        st.subheader("Readability Score Distribution")
        fig2 = px.histogram(analyzed, x="description_readability_score")
        fig2.update_traces(marker=dict(line=dict(color="white", width=2)))
        st.plotly_chart(fig2, use_container_width=True, key="desc_readability_chart")

        csv = analyzed.to_csv(index=False)
        st.download_button("Download Results", csv, "description_analysis.csv")

    st.markdown("---")

    # Single text analyzer
    st.subheader("Analyze a Single Description")

    with st.form(key="single_description_form"):
        text = st.text_area("Enter description:", height=150)
        submit_button = st.form_submit_button(
            "Analyze",
            type="primary",
            use_container_width=True,
        )

    if submit_button and text:
        emo = analyze_emotion(text)
        read = analyze_readability(text)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", emo["sentiment"])
            st.metric("Polarity", f"{emo['polarity']:.3f}")
            st.metric("Subjectivity", f"{emo['subjectivity']:.3f}")

        with col2:
            st.metric("Reading Level", read["category"])
            st.metric("Score", f"{read['score']:.1f}")
    elif submit_button and not text:
        st.warning("Please enter a description to analyze.")

# =========================
# Sidebar (pipeline actions)
# =========================
with st.sidebar:
    st.header("Pipeline")
    st.caption("Fetch → Enrich/Score → Explore")
    fetch_btn = st.button("🔄 Fetch TMDB Reviews (append)", use_container_width=True)
    score_btn = st.button(
        "⚙️ Enrich + Score (VADER + spaCy)", use_container_width=True
    )

    st.markdown("---")
    st.header("spaCy Evaluation")
    if st.button("Run Eval on data/test_reviews.csv", use_container_width=True):
        if os.path.exists("data/test_reviews.csv"):
            try:
                m = evaluateSpacy(
                    "data/test_reviews.csv",
                    textCol="text",
                    labelCol="label",
                    modelPath=SPACY_MODEL_PATH,
                )
                st.success(
                    f"Accuracy: {m['accuracy']:.3f} | Macro-F1: {m['macro_f1']:.3f}"
                )
                with st.expander("Confusion matrix"):
                    st.write("Labels:", m["confusion_matrix_labels"])
                    st.write(m["confusion_matrix"])
            except Exception as e:
                st.error(f"Eval failed: {e}")
        else:
            st.warning("data/test_reviews.csv not found.")

    st.markdown("---")
    st.caption(
        "Tip: set your TMDB API key in `.streamlit/secrets.toml` or TMDB_API_KEY env."
    )


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
        runScript(
            f'python "scripts/fetch_tmdb_reviews.py" '
            f'--netflix "{NETFLIX_PATH}" --output "{REVIEWS_PATH}" --limit 300'
        )

if score_btn:
    if not os.path.exists(NETFLIX_PATH):
        st.error("Missing data/netflix_titles.csv.")
    elif not os.path.exists(REVIEWS_PATH):
        st.error(
            "Missing data/reviews_raw.csv — click the sidebar **Fetch TMDB Reviews** first."
        )
    else:
        runScript(
            f'python "scripts/enrich_and_score.py" '
            f'--netflix "{NETFLIX_PATH}" --reviews "{REVIEWS_PATH}" '
            f'--output "{ENRICHED_PATH}" --spacyModel "{SPACY_MODEL_PATH}"'
        )

# =========================
# Tabs
# =========================
st.title("🎬 Netflix Sentiment Workbench (VADER + spaCy)")

tabs = st.tabs(
    [
        "📊 Overview",
        "🔎 Explore",
        "⚖️ NLP",
        "🧭 Title Explorer",
        "🎯 Recommender Engine",
        "📈 Statistics",
    ]
)

# ---------- Overview (README-style) ----------
with tabs[0]:
    st.markdown("# 📘 Netflix Sentiment Workbench")
    st.markdown(
        """
    Welcome to the **Netflix Sentiment Workbench**, an interactive data exploration tool that combines:

    - ⭐ **VADER** for rule-based sentiment scoring  
    - 🤖 **spaCy Text Classification** for machine-learned sentiment  
    - 🎬 **Netflix metadata** (titles, genres, release years)  
    - 📝 **User-generated TMDB reviews**  
    - 🔍 **A lightweight recommender system**  
    - 📊 **Rich visualizations** for patterns in content and sentiment  

    This app gives you a complete hands-on environment for exploring how sentiment varies across Netflix titles and how different NLP models compare.
    """
    )

    st.markdown("---")

    st.markdown("## 🚀 How the Pipeline Works")

    st.markdown(
        """
    **1️⃣ Fetch TMDB Reviews**  
    The pipeline calls the TMDB API to download user reviews for Netflix titles.  
    These reviews are appended to `data/reviews_raw.csv`.

    **2️⃣ Enrich + Score**  
    The script `enrich_and_score.py` merges Netflix metadata with reviews and creates a combined text field (`nlp_text`).  
    Each title is scored using:  
    - **VADER** → compound score + sentiment label  
    - **spaCy TextCat** → POSITIVE / NEGATIVE classification + probability  

    The processed dataset is stored in:

    ```text
    data/netflix_enriched_scored.csv
    ```

    **3️⃣ Explore the Dataset**  
    Filters allow you to analyze sentiment trends by:
    - Year  
    - Type (Movie / TV Show)  
    - Genres  
    - Model confidence  
    """
    )

    st.markdown("---")

    st.markdown("## 🧭 App Navigation Guide")

    st.markdown(
        """
    ### **📊 Overview**  
    You're here! This tab explains the purpose of the application and introduces its capabilities.

    ### **🔎 Explore**  
    Filter titles by:
    - Year  
    - Type  
    - spaCy sentiment  
    - Genre  
    - POSITIVE probability threshold  

    View trends and compare model outputs visually. This tab also contains core Netflix content visualizations.

    ### **🧠 NLP Sentiment Lab**  
    Compare VADER and spaCy predictions, tweak thresholds, inspect where they disagree, and view sentiment
    leaderboards by title and genre.

    ### **🧭 Title Explorer**  
    Search for any show or movie and view:
    - TMDB poster  
    - Sentiment details  
    - Combined review text  
    - Metadata and NLP insights  

    ### **🎯 Recommender Engine**  
    A keyword- and genre-based recommender that lets you find similar Netflix titles,
    plus sentiment-based content clustering.

    ### **Other Tabs**  
    - **📈 Statistics**: Additional data views  
    - **📝 Description Analyzer**: Analyze arbitrary description text  
    """
    )

    st.markdown("---")

    df_over = loadEnriched(ENRICHED_PATH)
    if df_over is not None and not df_over.empty:
        st.markdown("## 📈 Dataset Summary")
        total_titles = len(df_over)
        pos_rate = (df_over["spacy_label"] == "POSITIVE").mean() if "spacy_label" in df_over else 0.0
        avg_vader = df_over["vader_compound"].mean() if "vader_compound" in df_over else 0.0

        c1, c2, c3 = st.columns(3)
        with c1:
            kpiCard("Titles Scored", f"{total_titles:,}")
        with c2:
            kpiCard("spaCy % Positive", f"{pos_rate * 100:,.1f}%")
        with c3:
            kpiCard("Avg VADER Compound", f"{avg_vader:,.3f}")

        st.markdown("### 📉 Sentiment Distribution")
        left, right = st.columns(2)
        with left:
            try:
                fig = plotLabelCounts(df_over, which="spacy_label")
                st.plotly_chart(fig, use_container_width=True, key="overview_label_counts")
            except Exception as e:
                st.warning(f"Chart error: {e}")
        with right:
            try:
                scatter = plotVaderVsSpacy(df_over, textCol="nlp_text")
                st.plotly_chart(scatter, use_container_width=True, key="overview_vader_spacy")
            except Exception as e:
                st.warning(f"Chart error: {e}")
    else:
        st.info("No enriched dataset yet. Use the **sidebar pipeline buttons** to create it.")

# ---------- Explore ----------
with tabs[1]:
    df = loadEnriched(ENRICHED_PATH)
    if df is None or df.empty:
        st.info("Create an enriched dataset first.")
    else:
        st.subheader("Sentiment Filters")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            year_opt = ["(all)"] + getUniqueSorted(df["release_year"])
            year = st.selectbox("Release year", year_opt)
        with c2:
            types_opt = ["(all)"] + getUniqueSorted(df["type"])
            kind = st.selectbox("Type", types_opt)
        with c3:
            spacy_label = st.selectbox(
                "spaCy label", ["(all)", "POSITIVE", "NEGATIVE"]
            )
        with c4:
            min_conf = st.slider("Min spaCy POS prob", 0.0, 1.0, 0.0, 0.01)

        # Genres (multi-select) if present
        genres_flat = (
            sorted({g for lst in df["genres_list"] for g in lst})
            if "genres_list" in df.columns
            else []
        )
        selected_genres = st.multiselect("Genres (any)", genres_flat)

        mask = pd.Series(True, index=df.index)
        if year != "(all)":
            mask &= df["release_year"] == year
        if kind != "(all)":
            mask &= df["type"] == kind
        if spacy_label != "(all)":
            mask &= df["spacy_label"] == spacy_label
        if min_conf > 0:
            mask &= df["spacy_pos_prob"] >= min_conf
        if selected_genres:
            mask &= df["genres_list"].apply(
                lambda lst: any(g in lst for g in selected_genres)
            )

        view = df[mask].copy()
        st.caption(f"Showing {len(view):,} of {len(df):,} rows")

        # Pretty table
        show_cols = [
            "title",
            "type",
            "release_year",
            "spacy_label",
            "spacy_pos_prob",
            "vader_label",
            "vader_compound",
        ]
        if "listed_in" in view.columns:
            show_cols.append("listed_in")
        st.dataframe(view[show_cols].head(400))

        # Charts
        st.markdown("### Sentiment Charts")
        colA, colB = st.columns(2)
        with colA:
            try:
                fig1 = plotVaderVsSpacy(view, textCol="nlp_text")
                st.plotly_chart(
                    fig1, use_container_width=True, key="explore_vader_spacy"
                )
            except Exception as e:
                st.warning(f"Chart error: {e}")
        with colB:
            try:
                fig2 = plotLabelCounts(view, which="spacy_label")
                st.plotly_chart(
                    fig2, use_container_width=True, key="explore_label_counts"
                )
            except Exception as e:
                st.warning(f"Chart error: {e}")

        st.download_button(
            "Download current filtered view as CSV",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="filtered_view.csv",
            use_container_width=True,
        )

    # ---- Netflix content visualizations (base CSV) ----
    st.markdown("---")
    st.header("📊 Netflix Content Visualizations")

    df_raw = loadCsv(NETFLIX_PATH)
    if df_raw is None or df_raw.empty:
        st.info("Base Netflix CSV not found or empty.")
    else:
        df_clean, _ = cleanNetflixData(df_raw)
        df_clean = add_genres_list(df_clean)

        # Ratings table with year slider
        st.subheader("🎬 Ratings Table")
        year_cutoff = st.slider(
            "Minimum Release Year", min_value=1980, max_value=2025, value=2016
        )
        show_rating_table(df_clean, year=year_cutoff)

        # Top genres by country
        st.subheader("🌍 Top Genres by Country Over Time")
        country = st.text_input("Enter a country", value="United States")
        top_n_genres = st.slider("Top N Genres", 3, 10, 5)
        plot_top_genres_by_country(df_clean, country=country, top_n=top_n_genres)

    # ---- Interactive visualizations using enriched data ----
    df_viz = loadEnriched(ENRICHED_PATH)
    if df_viz is not None and not df_viz.empty:
        st.markdown("---")
        st.subheader("🎨 Interactive Visualizations")

        # Genre analysis
        st.subheader("🎭 Genre Analysis")
        if "primary_genre" in df_viz.columns:
            genre_counts = df_viz["primary_genre"].value_counts().head(15)
            fig_genres = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation="h",
                title="Top Genres / Runtime Buckets",
            )
            st.plotly_chart(fig_genres, use_container_width=True, key="viz_primary_genres")

            if "content_minutes" in df_viz.columns:
                genre_runtime = (
                    df_viz[["primary_genre", "content_minutes"]]
                    .dropna()
                    .groupby("primary_genre")["content_minutes"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(15)
                )
                if not genre_runtime.empty:
                    fig_runtime = px.bar(
                        x=genre_runtime.values,
                        y=genre_runtime.index,
                        orientation="h",
                        title="Average Content Minutes by Genre",
                    )
                    st.plotly_chart(
                        fig_runtime, use_container_width=True, key="viz_genre_runtime"
                    )
        else:
            st.info(
                "Genre metadata unavailable; showing runtime and region trends instead."
            )

        # Country distribution
        if "country" in df_viz.columns:
            all_countries = []
            for countries in df_viz["country"].dropna():
                if isinstance(countries, list):
                    all_countries.extend([c for c in countries if c])
                elif isinstance(countries, str):
                    all_countries.extend(
                        [c.strip() for c in countries.split(",") if c]
                    )

            if all_countries:
                country_series = (
                    pd.Series(all_countries, dtype="object")
                    .astype(str)
                    .str.strip()
                )
                exclusions = {"", "unknown", "nan", "none", "n/a"}
                country_series = country_series[
                    ~country_series.str.casefold().isin(exclusions)
                ]
                if not country_series.empty:
                    country_counts = country_series.value_counts().head(15)
                    fig_countries = px.bar(
                        x=country_counts.values,
                        y=country_counts.index,
                        orientation="h",
                        title="Top 15 Countries by Content Count",
                    )
                    st.plotly_chart(
                        fig_countries, use_container_width=True, key="viz_countries"
                    )
                else:
                    st.info("Country information not available for visualization.")
            else:
                st.info("Country information not available for visualization.")

        # Release year vs Rating (content rating, not stars)
        if "release_year" in df_viz.columns and "rating" in df_viz.columns:
            trend_df = df_viz.copy()
            trend_df["release_year"] = pd.to_numeric(
                trend_df["release_year"], errors="coerce"
            )
            trend_df = trend_df.dropna(subset=["release_year", "rating"])

            if not trend_df.empty:
                trend_df["release_year"] = trend_df["release_year"].astype(int)
                rating_counts = (
                    trend_df.groupby(["release_year", "rating"])
                    .size()
                    .reset_index(name="count")
                )
                rating_counts["rating"] = rating_counts["rating"].astype(str)
                rating_counts = rating_counts[
                    ~rating_counts["rating"]
                    .str.casefold()
                    .isin({"unknown", "nan", "none", ""})
                ]
                if not rating_counts.empty:
                    rating_counts = rating_counts.sort_values(
                        ["release_year", "rating"]
                    )
                    fig_rating_trend = px.line(
                        rating_counts,
                        x="release_year",
                        y="count",
                        color="rating",
                        markers=True,
                        title="Content Rating Trend by Release Year",
                        labels={
                            "count": "Number of Titles",
                            "release_year": "Release Year",
                        },
                    )
                    st.plotly_chart(
                        fig_rating_trend,
                        use_container_width=True,
                        key="viz_rating_trend",
                    )
                else:
                    st.info("Rating trend chart skipped: no rating data available.")
            else:
                st.info("Insufficient release year data for rating trends.")

        # VADER score distribution
        if "vader_compound" in df_viz.columns:
            st.subheader("📊 VADER Score Distribution")

            fig_vader = px.histogram(
                df_viz,
                x="vader_compound",
                nbins=50,
                title="Distribution of VADER Compound Scores",
                labels={
                    "vader_compound": "VADER Compound Score",
                    "count": "Number of Titles",
                },
            )
            fig_vader.update_traces(
                marker=dict(color="#1f77b4", line=dict(color="white", width=1))
            )
            st.plotly_chart(fig_vader, use_container_width=True, key="viz_vader_dist")

# ---------- NLP ----------
with tabs[2]:
    nlp_tabs = st.tabs(
        ["⚖️ Model Compare", "📝 Description Analyzer", "🏆 Top Titles", "📊 Genre Summary"]
    )

    # Model Compare sub-tab
    with nlp_tabs[0]:
        df = loadEnriched(ENRICHED_PATH)
        if df is None or df.empty:
            st.info("Create an enriched dataset first.")
        else:
            st.subheader("Agreement & Thresholds")
            c1, c2 = st.columns(2)
            with c1:
                pos_thresh = st.slider(
                    "VADER positive threshold", 0.0, 0.5, 0.05, 0.01
                )
            with c2:
                neg_thresh = st.slider(
                    "VADER negative threshold", -0.5, 0.0, -0.05, 0.01
                )

            # recompute VADER label on the fly for comparison
            vlab = pd.Series("NEUTRAL", index=df.index)
            vlab = vlab.mask(df["vader_compound"] >= pos_thresh, "POSITIVE")
            vlab = vlab.mask(df["vader_compound"] <= neg_thresh, "NEGATIVE")

            agree = vlab == df["spacy_label"]
            st.write(f"**Agreement rate:** {(agree.mean() * 100):.1f}%  (n={len(df)})")

            # Show disagreements table
            st.markdown("#### Disagreements")
            dis = df[~agree].copy()
            st.dataframe(
                dis[
                    [
                        "title",
                        "type",
                        "release_year",
                        "spacy_label",
                        "spacy_pos_prob",
                        "vader_compound",
                        "vader_label",
                    ]
                ].head(300)
            )

            st.markdown("#### VADER vs spaCy")
            try:
                scatter = plotVaderVsSpacy(df, textCol="nlp_text")
                st.plotly_chart(
                    scatter,
                    use_container_width=True,
                    key="model_compare_vader_spacy",
                )
            except Exception as e:
                st.warning(f"Chart error: {e}")

    # Description Analyzer sub-tab
    with nlp_tabs[1]:
        description_analyzer_tab()

    # Top Titles sub-tab (moved from Sentiment Summary by Genre tab)
    with nlp_tabs[2]:
        df = loadEnriched(ENRICHED_PATH)
        st.header("🏆 Top Positive & Negative Titles")
        if df is None or df.empty:
            st.info(
                "No enriched dataset yet. Use the **sidebar pipeline buttons** to create it."
            )
        else:
            col1, col2, col3 = st.columns([2, 2, 1])

            # Filters
            with col1:
                genres_opts = (
                    sorted({g for lst in df.get("genres_list", []) for g in lst})
                    if "genres_list" in df.columns
                    else []
                )
                selected_genres = st.multiselect("Select genres", genres_opts)
            with col2:
                type_opts = (
                    ["(all)"] + sorted(df["type"].dropna().unique())
                    if "type" in df.columns
                    else ["(all)"]
                )
                selected_type = st.selectbox("Type filter", type_opts)
            with col3:
                max_titles = (
                    int(df["title"].nunique())
                    if "title" in df.columns
                    else len(df)
                )
                top_n = st.slider(
                    "Top N titles",
                    min_value=1,
                    max_value=max(1, max_titles),
                    value=min(10, max(1, max_titles)),
                )

            # Base frame
            s = df.copy()
            s["spacy_pos_prob"] = s.get(
                "spacy_pos_prob", pd.Series([0.0] * len(s))
            ).fillna(0.0)
            s["vader_compound"] = s.get(
                "vader_compound", pd.Series([0.0] * len(s))
            ).fillna(0.0)
            s["vader_norm"] = (s["vader_compound"] + 1.0) / 2.0
            s["combined_score"] = s["spacy_pos_prob"] + s["vader_norm"]

            # Apply filters
            if selected_genres and "genres_list" in s.columns:
                s = s[
                    s["genres_list"].apply(
                        lambda lst: any(g in lst for g in selected_genres)
                    )
                ]
            if selected_type != "(all)":
                s = s[s["type"] == selected_type]

            # Collapse to one row per title, preferring more reviews
            if "review_join" in s.columns:
                s["review_count"] = (
                    s["review_join"]
                    .fillna("")
                    .astype(str)
                    .apply(lambda x: 0 if not x else x.count(" || ") + 1)
                )
                s = (
                    s.sort_values("review_count", ascending=False)
                    .drop_duplicates(subset=["title"], keep="first")
                )
            else:
                s = s.drop_duplicates(subset=["title"], keep="first")

            # Positive / Negative sets
            pos = s.sort_values("combined_score", ascending=False).head(top_n)
            neg = s.sort_values("combined_score", ascending=True).head(top_n)

            # Helper to truncate titles for display
            def truncate_title(title: str, length: int = 30) -> str:
                title = str(title)
                return title if len(title) <= length else title[:length] + "..."

            pos = pos.copy()
            neg = neg.copy()
            pos["title_trunc"] = pos["title"].apply(truncate_title)
            neg["title_trunc"] = neg["title"].apply(truncate_title)

            # Positive Titles Chart
            figp = px.bar(
                pos.sort_values("combined_score", ascending=True),
                x="combined_score",
                y="title_trunc",
                orientation="h",
                color="combined_score",
                color_continuous_scale="Greens",
                labels={"combined_score": "Score", "title_trunc": "Title"},
                hover_data={
                    "title": True,
                    "type": True,
                    "release_year": True,
                    "spacy_pos_prob": True,
                    "vader_compound": True,
                },
            )
            figp.update_layout(yaxis={"categoryorder": "total ascending"})

            # Negative Titles Chart
            fign = px.bar(
                neg.sort_values("combined_score", ascending=True),
                x="combined_score",
                y="title_trunc",
                orientation="h",
                color="combined_score",
                color_continuous_scale="Reds",
                labels={"combined_score": "Score", "title_trunc": "Title"},
                hover_data={
                    "title": True,
                    "type": True,
                    "release_year": True,
                    "spacy_pos_prob": True,
                    "vader_compound": True,
                },
            )
            fign.update_layout(yaxis={"categoryorder": "total ascending"})

            # Display charts and tables
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("Top Positive Titles")
                st.plotly_chart(figp, use_container_width=True)
                st.dataframe(
                    pos[
                        [
                            "title",
                            "type",
                            "release_year",
                            "spacy_pos_prob",
                            "vader_compound",
                            "combined_score",
                        ]
                    ],
                    use_container_width=True,
                )
                st.download_button(
                    "Download Positive CSV",
                    data=pos.to_csv(index=False).encode("utf-8"),
                    file_name="top_positive_titles.csv",
                )

            with col_right:
                st.subheader("Top Negative Titles")
                st.plotly_chart(fign, use_container_width=True)
                st.dataframe(
                    neg[
                        [
                            "title",
                            "type",
                            "release_year",
                            "spacy_pos_prob",
                            "vader_compound",
                            "combined_score",
                        ]
                    ],
                    use_container_width=True,
                )
                st.download_button(
                    "Download Negative CSV",
                    data=neg.to_csv(index=False).encode("utf-8"),
                    file_name="top_negative_titles.csv",
                )

    # Genre Sentiment Summary sub-tab
    with nlp_tabs[3]:
        df = loadEnriched(ENRICHED_PATH)
        st.header("📊 Sentiment Summary by Genre")
        if df is None or df.empty:
            st.info(
                "No enriched dataset yet. Use the **sidebar pipeline buttons** to create it."
            )
        else:
            type_opts = (
                ["(all)"] + sorted(df["type"].dropna().unique())
                if "type" in df.columns
                else ["(all)"]
            )
            selected_type = st.selectbox("Filter by Type", type_opts)

            s2 = df.copy()
            if selected_type != "(all)":
                s2 = s2[s2["type"] == selected_type]

            if "genres_list" not in s2.columns:
                st.warning("No genre information found in dataset.")
            else:
                s2 = s2.explode("genres_list")

                # Combined sentiment score
                s2["avg_sentiment"] = s2["spacy_pos_prob"].fillna(0.0) + (
                    s2["vader_compound"].fillna(0.0) + 1.0
                ) / 2.0

                genre_summary = (
                    s2.groupby("genres_list")
                    .agg(
                        avg_sentiment=("avg_sentiment", "mean"),
                        num_titles=("title", "count"),
                    )
                    .reset_index()
                    .sort_values("num_titles", ascending=False)
                )

                def sentiment_label(score: float) -> str:
                    if score > 0.55:
                        return "Positive"
                    elif score < 0.45:
                        return "Negative"
                    else:
                        return "Neutral"

                genre_summary["Sentiment"] = genre_summary["avg_sentiment"].apply(
                    sentiment_label
                )

                if len(genre_summary) > 0:
                    top_n_genres = st.slider(
                        "Top N genres to show",
                        min_value=1,
                        max_value=len(genre_summary),
                        value=min(10, len(genre_summary)),
                    )
                    genre_summary_top = genre_summary.head(top_n_genres)

                    st.dataframe(genre_summary_top, use_container_width=True)

                    fig = px.bar(
                        genre_summary_top,
                        x="avg_sentiment",
                        y="genres_list",
                        color="Sentiment",
                        color_discrete_map={
                            "Positive": "#2ca02c",
                            "Neutral": "#ffbb78",
                            "Negative": "#d62728",
                        },
                        orientation="h",
                        labels={
                            "avg_sentiment": "Average Sentiment",
                            "genres_list": "Genre",
                        },
                        title="Average Sentiment by Genre",
                    )
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No genre records available after filtering.")

# ---------- Title Explorer ----------
with tabs[3]:
    df = loadEnriched(ENRICHED_PATH)
    if df is None or df.empty:
        st.info("Create an enriched dataset first.")
    else:
        st.subheader("Find a title")
        query = st.text_input("Search title", placeholder="Start typing…")
        if query:
            sub = df[df["title"].str.contains(query, case=False, na=False)].head(
                25
            ).copy()
        else:
            sub = df.head(25).copy()

        for _, row in sub.iterrows():
            with st.container(border=True):
                top = st.columns([1, 3, 2])
                # Poster
                with top[0]:
                    poster = fetchPosterPath(
                        row["title"], row.get("release_year"), row.get("type", "movie")
                    )
                    if poster:
                        st.image(f"{TMDB_IMAGE_BASE}{poster}")
                    else:
                        st.write("No image")
                # Meta
                with top[1]:
                    st.subheader(str(row["title"]))
                    meta = f"{row.get('type', '?')} • {row.get('release_year', '?')}"
                    st.caption(meta)
                    # badges
                    badge(
                        f"spaCy: {row.get('spacy_label', '?')}",
                        "positive"
                        if row.get("spacy_label") == "POSITIVE"
                        else "negative",
                    )
                    st.write(
                        f"spaCy pos prob: {row.get('spacy_pos_prob', 0.0):.3f}"
                    )
                    badge(
                        f"VADER: {row.get('vader_label', '?')}",
                        "positive"
                        if row.get("vader_label") == "POSITIVE"
                        else (
                            "negative"
                            if row.get("vader_label") == "NEGATIVE"
                            else "neutral"
                        ),
                    )
                    st.write(
                        f"VADER compound: {row.get('vader_compound', 0.0):.3f}"
                    )
                # Text
                with top[2]:
                    txt = str(row.get("nlp_text", "")).strip()
                    if len(txt) > 280:
                        st.text_area(
                            "Text (truncated)",
                            value=txt[:800] + ("…" if len(txt) > 800 else ""),
                            height=160,
                        )
                        with st.expander("Show full text"):
                            st.write(txt)
                    else:
                        st.text_area("Text", value=txt, height=160)

# ---------- Recommender Search Engine ----------
with tabs[4]:
    # Load Data
    df_raw = loadCsv(NETFLIX_PATH)
    if df_raw is None or df_raw.empty:
        st.info("Base Netflix CSV not found or empty.")
    else:
        df_clean, _ = cleanNetflixData(df_raw)
        df_clean = add_genres_list(df_clean)

        st.subheader("🎯 Recommender Search Engine")
        st.write(
            "Write keywords and select genres you are interested in, "
            "and choose if you want **all** or **any** of these in the search."
        )

        col1, col2 = st.columns(2)
        with col1:
            # User inputs keywords
            keywords_input = st.text_input("Keywords (comma-separated)", value="school")
            keyword_mode = st.radio("Keyword Match Mode", ["any", "all"])
        with col2:
            selected_genres = st.multiselect(
                "Genres",
                options=sorted(
                    set(g for sublist in df_clean["genres_list"] for g in sublist if g)
                ),
            )
            genre_mode = st.radio("Genre Match Mode", ["any", "all"])

        top_n = st.slider("Number of Recommendations", 1, 20, 10)
        st.write("""The fuzzy match slider controls how strictly keywords must match the movie descriptions.
             - **Higher values (closer to 100)** → only very close matches are included (e.g., 'color' matches 'color').
             - **Lower values (closer to 50)** → looser matches are allowed (e.g., 'color' also matches 'colour' or 'colr').""")
        
        # Fuzzy match threshold slider
        fuzzy_threshold = st.slider("Fuzzy match threshold", min_value=50, max_value=100, value=90)

        st.subheader("Recommended Titles")
        if st.button("Run Recommender"):
            keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
            results = run_recommender(
                df_clean, 
                keywords, 
                selected_genres, 
                top_n, 
                keyword_mode, 
                genre_mode, 
                fuzzy_threshold=fuzzy_threshold)
            if results.empty:
                st.warning("No matches found. Try different keywords or genres.")
            else:
                st.dataframe(results.reset_index(drop=True), use_container_width=True)

    # ---------- Sentiment-based Content Clustering ----------
    st.markdown("---")
    st.subheader("🎯 Content Clustering (Sentiment Space)")

    df_enriched = loadEnriched(ENRICHED_PATH)
    if df_enriched is None or df_enriched.empty:
        st.info("Run the sentiment pipeline (sidebar) to enable clustering.")
    else:
        if (
            "vader_compound" in df_enriched.columns
            and "spacy_pos_prob" in df_enriched.columns
            and "release_year" in df_enriched.columns
        ):
            st.write("We cluster titles using VADER, spaCy, and release year.")

            cluster_df = df_enriched.copy()
            cluster_df["release_year_numeric"] = pd.to_numeric(
                cluster_df["release_year"], errors="coerce"
            )

            features_df = cluster_df[
                ["vader_compound", "spacy_pos_prob", "release_year_numeric"]
            ].dropna()

            if len(features_df) > 10:
                n_clusters = st.slider(
                    "Number of clusters:",
                    2,
                    8,
                    4,
                    key="recommender_n_clusters",
                )

                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_df)

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features_scaled)

                features_df = features_df.copy()
                features_df["cluster"] = cluster_labels
                cluster_df.loc[features_df.index, "cluster_label"] = cluster_labels

                col1, col2 = st.columns(2)

                with col1:
                    cluster_counts = (
                        pd.Series(cluster_labels).value_counts().sort_index()
                    )
                    fig_clusters = px.bar(
                        x=cluster_counts.index,
                        y=cluster_counts.values,
                        title="Content Clusters Distribution",
                        labels={"x": "Cluster", "y": "Number of Titles"},
                    )
                    st.plotly_chart(
                        fig_clusters,
                        use_container_width=True,
                        key="recommender_clusters_bar",
                    )

                with col2:
                    fig_scatter = px.scatter(
                        features_df,
                        x="vader_compound",
                        y="spacy_pos_prob",
                        color=features_df["cluster"].astype(str),
                        title="Clusters in Sentiment Space",
                        labels={
                            "vader_compound": "VADER Compound",
                            "spacy_pos_prob": "spaCy Positive Prob",
                            "color": "Cluster",
                        },
                    )
                    st.plotly_chart(
                        fig_scatter,
                        use_container_width=True,
                        key="recommender_clusters_scatter",
                    )

                st.subheader("📊 Cluster Statistics")
                cluster_stats = (
                    features_df.groupby("cluster")
                    .agg(
                        vader_mean=("vader_compound", "mean"),
                        vader_std=("vader_compound", "std"),
                        spacy_mean=("spacy_pos_prob", "mean"),
                        spacy_std=("spacy_pos_prob", "std"),
                        year_mean=("release_year_numeric", "mean"),
                        year_min=("release_year_numeric", "min"),
                        year_max=("release_year_numeric", "max"),
                    )
                    .round(3)
                )
                st.dataframe(cluster_stats, use_container_width=True)

                st.subheader("📋 Sample Titles by Cluster")
                selected_cluster = st.selectbox(
                    "Select cluster to view sample titles:",
                    options=sorted(features_df["cluster"].unique()),
                    key="recommender_cluster_selector",
                )

                cluster_indices = features_df[
                    features_df["cluster"] == selected_cluster
                ].index
                cluster_sample = df_enriched.loc[cluster_indices].head(10)

                display_cols = ["title"]
                if "type" in cluster_sample.columns:
                    display_cols.append("type")
                if "release_year" in cluster_sample.columns:
                    display_cols.append("release_year")
                if "spacy_label" in cluster_sample.columns:
                    display_cols.append("spacy_label")
                if "vader_compound" in cluster_sample.columns:
                    display_cols.append("vader_compound")

                st.dataframe(cluster_sample[display_cols], use_container_width=True)
            else:
                st.warning("Not enough data points for clustering (need at least 10 full rows).")
        else:
            st.info(
                "Clustering requires vader_compound, spacy_pos_prob, and release_year columns in the enriched dataset."
            )

# ---------- Statistics DK ----------
with tabs[5]:
    df = loadEnriched(ENRICHED_PATH)
    if df is None or df.empty:
        st.info(
            "No enriched dataset yet. Please create the enriched dataset first (sidebar pipeline)."
        )
    else:
        st.subheader("📈 Statistical Analysis DK")

        # Time Series in Stats DK (moved from Time Series tab)
        st.subheader("⏰ Time Series - Content by Release Year")
        if "release_year" in df.columns:
            release_years = (
                pd.to_numeric(df["release_year"], errors="coerce")
                .dropna()
                .astype(int)
            )
            if not release_years.empty:
                year_counts = release_years.value_counts().sort_index()

                fig_trend = px.line(
                    x=year_counts.index,
                    y=year_counts.values,
                    title="Netflix Content by Release Year",
                    labels={"x": "Release Year", "y": "Number of Titles"},
                )
                fig_trend.update_traces(line=dict(color="#e50914", width=2))
                st.plotly_chart(
                    fig_trend,
                    use_container_width=True,
                    key="stats_timeseries_releases",
                )
            else:
                st.info("Release year data not available for time series analysis.")

        # Try to load and merge runtime data if available
        runtime_path = "netflix_movies_tv_runtime.csv"
        if os.path.exists(runtime_path):
            try:
                runtime_df = pd.read_csv(runtime_path)
                if "title" in df.columns and "title" in runtime_df.columns:
                    df = df.merge(
                        runtime_df[["title", "rating_stars"]],
                        on="title",
                        how="left",
                        suffixes=("", "_runtime"),
                    )
            except Exception as e:
                st.warning(f"Could not load runtime data: {e}")

        # Release Year stats
        st.subheader("🎬 Release Year Statistics")
        if "release_year" in df.columns:
            release_year = pd.to_numeric(df["release_year"], errors="coerce").dropna()
            if not release_year.empty:
                year_stats = release_year.describe()

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Release Year Summary**")
                    st.dataframe(year_stats.round(0))
                with col2:
                    fig_year_hist = px.histogram(
                        release_year,
                        nbins=20,
                        title="Release Year Distribution",
                        labels={
                            "value": "Release Year",
                            "count": "Number of Titles",
                        },
                    )
                    fig_year_hist.update_traces(
                        marker=dict(line=dict(color="white", width=1))
                    )
                    st.plotly_chart(
                        fig_year_hist,
                        use_container_width=True,
                        key="stats_year_hist",
                    )
            else:
                st.info("Release year data not available after processing.")

        # Rating distribution
        st.subheader("⭐ Rating Analysis (TMDB-style ratings)")
        rating_col = None

        for col in df.columns:
            if "rating" in col.lower() and (
                "_runtime" in col.lower() or "stars" in col.lower()
            ):
                try:
                    test_vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(test_vals) > 0 and test_vals.max() <= 10:
                        rating_col = col
                        break
                except Exception:
                    continue

        if not rating_col:
            possible_rating_cols = [
                "rating_stars",
                "rating_stars_runtime",
                "rating_runtime",
                "total_rating_runtime",
                "total_rating",
                "vote_average",
                "tmdb_rating",
            ]
            for col in possible_rating_cols:
                if col in df.columns:
                    rating_col = col
                    break

        if rating_col:
            try:
                ratings = pd.to_numeric(df[rating_col], errors="coerce").dropna()

                if len(ratings) > 0:
                    fig_rating = px.histogram(
                        x=ratings,
                        nbins=20,
                        title="TMDB Rating Distribution (0-10 scale)",
                        labels={"x": "Rating (0-10)", "y": "Number of Titles"},
                    )
                    fig_rating.update_traces(
                        marker=dict(line=dict(color="white", width=1))
                    )
                    fig_rating.update_layout(
                        xaxis_title="Rating (0-10)",
                        yaxis_title="Number of Titles",
                        showlegend=False,
                    )
                    st.plotly_chart(
                        fig_rating,
                        use_container_width=True,
                        key="stats_rating_hist",
                    )

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
            with st.expander("🔍 Debug: Available columns"):
                st.write("Looking for TMDB rating columns. All columns with 'rating':")
                rating_cols = [col for col in df.columns if "rating" in col.lower()]
                st.write(rating_cols)
                st.write("\nAll columns:")
                st.write(list(df.columns))
            st.info(
                "No TMDB rating column found. Please ensure netflix_movies_tv_runtime.csv is loaded correctly."
            )

        # Sentiment Analysis Statistics
        st.subheader("😊 Sentiment Analysis Statistics")

        if "vader_compound" in df.columns or "spacy_pos_prob" in df.columns:
            col1, col2 = st.columns(2)

            # VADER Statistics
            if "vader_compound" in df.columns:
                with col1:
                    st.write("**VADER Compound Score Statistics**")
                    vader_scores = df["vader_compound"].dropna()

                    if not vader_scores.empty:
                        vader_stats = vader_scores.describe()
                        st.dataframe(vader_stats.round(3))

                        fig_vader = px.histogram(
                            x=vader_scores,
                            nbins=30,
                            title="VADER Compound Score Distribution",
                            labels={
                                "x": "VADER Compound Score",
                                "y": "Count",
                            },
                        )
                        fig_vader.update_traces(
                            marker=dict(
                                color="#564d4d",
                                line=dict(color="white", width=1),
                            )
                        )
                        st.plotly_chart(
                            fig_vader,
                            use_container_width=True,
                            key="stats_vader_dist",
                        )

                        if "vader_label" in df.columns:
                            vader_label_counts = df["vader_label"].value_counts()
                            fig_vader_labels = px.pie(
                                values=vader_label_counts.values,
                                names=vader_label_counts.index,
                                title="VADER Sentiment Distribution",
                                color=vader_label_counts.index,
                                color_discrete_map={
                                    "POSITIVE": "#2ca02c",
                                    "NEGATIVE": "#d62728",
                                    "NEUTRAL": "#ffbb78",
                                },
                            )
                            st.plotly_chart(
                                fig_vader_labels,
                                use_container_width=True,
                                key="stats_vader_labels",
                            )

            # spaCy Statistics
            if "spacy_pos_prob" in df.columns:
                with col2:
                    st.write("**spaCy Positive Probability Statistics**")
                    spacy_scores = df["spacy_pos_prob"].dropna()

                    if not spacy_scores.empty:
                        spacy_stats = spacy_scores.describe()
                        st.dataframe(spacy_stats.round(3))

                        fig_spacy = px.histogram(
                            x=spacy_scores,
                            nbins=30,
                            title="spaCy Positive Probability Distribution",
                            labels={
                                "x": "spaCy Positive Probability",
                                "y": "Count",
                            },
                        )
                        fig_spacy.update_traces(
                            marker=dict(
                                color="#17becf",
                                line=dict(color="white", width=1),
                            )
                        )
                        st.plotly_chart(
                            fig_spacy,
                            use_container_width=True,
                            key="stats_spacy_dist",
                        )

                        if "spacy_label" in df.columns:
                            spacy_label_counts = df["spacy_label"].value_counts()
                            fig_spacy_labels = px.pie(
                                values=spacy_label_counts.values,
                                names=spacy_label_counts.index,
                                title="spaCy Sentiment Distribution",
                                color=spacy_label_counts.index,
                                color_discrete_map={
                                    "POSITIVE": "#2ca02c",
                                    "NEGATIVE": "#d62728",
                                },
                            )
                            st.plotly_chart(
                                fig_spacy_labels,
                                use_container_width=True,
                                key="stats_spacy_labels",
                            )

            # Sentiment Correlation
            if "vader_compound" in df.columns and "spacy_pos_prob" in df.columns:
                st.write("**Sentiment Score Correlation**")
                correlation = (
                    df[["vader_compound", "spacy_pos_prob"]].corr().iloc[0, 1]
                )

                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.metric("Correlation Coefficient", f"{correlation:.3f}")

                    if abs(correlation) > 0.7:
                        st.success("Strong correlation")
                    elif abs(correlation) > 0.4:
                        st.info("Moderate correlation")
                    else:
                        st.warning("Weak correlation")

                with col_b:
                    sample_size = min(1000, len(df))
                    fig_corr = px.scatter(
                        df.sample(sample_size),
                        x="vader_compound",
                        y="spacy_pos_prob",
                        title="VADER vs spaCy Sentiment Correlation",
                        labels={
                            "vader_compound": "VADER Compound",
                            "spacy_pos_prob": "spaCy Pos Prob",
                        },
                        opacity=0.5,
                    )
                    st.plotly_chart(
                        fig_corr,
                        use_container_width=True,
                        key="stats_sentiment_corr",
                    )
        else:
            st.info("Sentiment analysis columns not found in the dataset.")
