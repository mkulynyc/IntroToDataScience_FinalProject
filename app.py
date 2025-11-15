import os
import sys
import time
from typing import List, Optional

import pandas as pd
import streamlit as st
import plotly.express as px

# Make sure we can import from src/
sys.path.append(os.path.abspath("src"))

# Your tab utilities (these are safe)
from src.utils.shared_utils import analyze_emotion, analyze_readability


# -------------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------------
st.set_page_config(page_title="Netflix Centralized App", layout="wide")

st.title("🎬 Netflix Centralized Group App")


# -------------------------------------------------------
# Your Description Analyzer Tab
# -------------------------------------------------------
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
    st.header("📝 Description Analyzer (Your Contribution)")

    st.write("Upload a CSV with a **description** column to analyze sentiment & readability.")

    uploaded = st.file_uploader("Upload CSV", type=['csv'])

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
            st.plotly_chart(fig1, use_container_width=True)

            # Readability chart
            st.subheader("Readability Score Distribution")
            fig2 = px.histogram(analyzed, x="description_readability_score")
            st.plotly_chart(fig2, use_container_width=True)

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


# -------------------------------------------------------
# Create Tabs
# -------------------------------------------------------
tabs = st.tabs([
    "📊 Overview",
    "🔍 Explore",
    "⚖️ Model Compare",
    "🧭 Title Explorer",
    "⚙️ Ingest & Score",
    "🎯 Recommender Engine",
    "📈 Visualizations",
    "📝 Description Analyzer"
])


# -------------------------------------------------------
# Other TABS (Lazy Loaded)
# -------------------------------------------------------
# ⭐⭐ These import ONLY inside their tab. If they fail, your tab still works.

# --------------------
# 1️⃣ Overview Tab
# --------------------
with tabs[0]:
    st.header("📊 Overview (Group Tab)")
    try:
        from data_load import loadEnriched
        from viz import plotVaderVsSpacy, plotLabelCounts

        df = loadEnriched("data/netflix_enriched_scored.csv")
        if df is None or df.empty:
            st.info("No enriched dataset found.")
        else:
            st.write("Dataset loaded. Showing basic statistics:")
            st.write(df.head())

            fig1 = plotLabelCounts(df, which="spacy_label")
            st.plotly_chart(fig1)
    except Exception as e:
        st.error("Overview tab failed to load due to missing modules.")
        st.warning(str(e))


# --------------------
# 2️⃣ Explore Tab
# --------------------
with tabs[1]:
    st.header("🔍 Explore (Group Tab)")
    st.info("This tab depends on group files. If they fail, it's okay — your tab still works.")
    try:
        from data_load import loadEnriched
        df = loadEnriched("data/netflix_enriched_scored.csv")
        if df is not None:
            st.dataframe(df.head())
    except Exception as e:
        st.error("Explore tab could not load.")
        st.warning(str(e))


# --------------------
# 3️⃣ Model Compare
# --------------------
with tabs[2]:
    st.header("⚖️ Model Compare (Group Tab)")
    try:
        from viz import plotVaderVsSpacy
        st.write("Loaded visualization tools.")
    except Exception as e:
        st.error("Model Compare tab failed.")
        st.warning(str(e))


# --------------------
# 4️⃣ Title Explorer
# --------------------
with tabs[3]:
    st.header("🧭 Title Explorer (Group Tab)")
    try:
        from data_load import loadEnriched
        df = loadEnriched("data/netflix_enriched_scored.csv")
        if df is not None:
            st.dataframe(df.head())
    except Exception as e:
        st.error("Title Explorer failed.")
        st.warning(str(e))


# --------------------
# 5️⃣ Ingest & Score
# --------------------
with tabs[4]:
    st.header("⚙️ Ingest & Score (Group Tab)")
    st.info("This depends on team scripts and spacy model.")
    try:
        from scripts.fetch_tmdb_reviews import fetch_reviews
    except Exception as e:
        st.error("Ingest tab failed.")
        st.warning(str(e))


# --------------------
# 6️⃣ Recommender Engine
# --------------------
with tabs[5]:
    st.header("🎯 Recommender Engine (Group Tab)")
    try:
        from engine import run_recommender
        st.write("Recommender engine ready.")
    except Exception as e:
        st.error("Recommender Engine tab failed.")
        st.warning(str(e))


# --------------------
# 7️⃣ Visualizations
# --------------------
with tabs[6]:
    st.header("📈 Visualizations (Group Tab)")
    try:
        from plots import show_rating_table
        st.write("Loaded visualization functions.")
    except Exception as e:
        st.error("Visualizations tab failed.")
        st.warning(str(e))


# --------------------
# 8️⃣ YOUR TAB
# --------------------
with tabs[7]:
    description_analyzer_tab()
