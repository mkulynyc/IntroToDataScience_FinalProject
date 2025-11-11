#!/usr/bin/env python3

"""Netflix Enhanced Description Analysis Dashboard

This module provides a Streamlit dashboard that shows enhanced NLP
analysis for Netflix descriptions. It expects an enhanced CSV with
precomputed fields (or will run an in-process enhancement on demand).
"""

from pathlib import Path
import json
import os
import sys
import warnings

import pandas as pd
import streamlit as st
import plotly.express as px

# Keep warnings quiet for a nicer UI
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure repository modules import correctly
current_dir = Path(__file__).parent.resolve()
src_dir = current_dir.parent  # Go up to src/
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import local analysis helpers
try:
    from analyzers.enhanced_nlp_description_analyzer import (
        generate_ai_summary,
        extract_global_keywords,
        enrich_netflix_with_enhanced_description_analysis,
    )
    from utils.shared_utils import analyze_emotion, analyze_readability, count_words
except Exception as exc:
    # Handle import errors gracefully when running syntax checks
    pass


def main():
    """Main Streamlit app entrypoint."""

    st.set_page_config(
        page_title="Enhanced Netflix Description Analyzer",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Basic styling
    st.markdown(
        """
    <style>
    .main-header { 
        font-size: 2.1rem; 
        font-weight: 700; 
        color: #E50914; 
        text-align: center; 
    }
    .metric-card { 
        background: #f8f9fa; 
        padding: 0.75rem; 
        border-radius: 6px; 
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">Enhanced Netflix Description Analyzer</div>', unsafe_allow_html=True)
    st.markdown("*Advanced NLP analysis with AI summaries, keyword extraction and readability metrics.*")

    with st.expander("What's enhanced about this version?"):
        st.markdown(
            """
        This version includes several advanced NLP features:
        - AI-powered summaries (optional transformer models)
        - TF-IDF based global keyword extraction
        - Readability and emotion metrics
        - Word count and summary fields

        It prefers a precomputed CSV `netflix_with_enhanced_description_analysis.csv` in the project's `data/` folder for speed.
            """
        )

    # Sidebar controls and data loading
    st.sidebar.title("Analysis Options")
    # Navigate from src/apps/ up to project root, then into data/
    repo_root = Path(__file__).parent.parent.parent
    data_dir = repo_root / "data"
    enhanced_file = data_dir / "netflix_with_enhanced_description_analysis.csv"

    df = None
    if enhanced_file.exists():
        try:
            df = pd.read_csv(enhanced_file)
            st.sidebar.success("Enhanced dataset loaded")
            st.sidebar.write(f"Shows: {len(df):,}")
        except Exception as exc:
            st.sidebar.error(f"Failed to read enhanced file: {exc}")
    else:
        st.sidebar.info("Precomputed enhanced dataset not found")

    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (must include 'description')", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.info(f"Using uploaded: {uploaded_file.name}")
        except Exception as exc:
            st.sidebar.error(f"Could not read uploaded CSV: {exc}")

    if df is None:
        st.info("No data available. Please provide a dataset or place the enhanced CSV in data/.")
        return

    # Determine whether the DataFrame already contains enhanced columns
    enhanced_cols = {
        "nlp_summary",
        "nlp_word_count",
        "description_emotion_polarity",
        "description_emotion_subjectivity",
        "description_emotion_sentiment",
        "description_readability_score",
        "description_readability_category",
        "nlp_global_keywords",
    }

    has_enhanced = enhanced_cols.issubset(set(df.columns))

    st.sidebar.write(f"Columns: {len(df.columns)}")
    st.sidebar.write(f"Has enhanced analysis: {has_enhanced}")

    if has_enhanced:
        enriched_df = df.copy()
        st.session_state["enriched_df"] = enriched_df
        show_results(enriched_df)
        return

    # Provide an action to run enhancement in-process
    st.header("Enhanced Description Analysis")
    if st.button("Run Enhanced Analysis"):
        with st.spinner("Running enhanced analysis — this may take a while for AI summaries"):
            enriched_df = enrich_netflix_with_enhanced_description_analysis(df.copy())
        st.success("Analysis complete")
        st.session_state["enriched_df"] = enriched_df
        show_results(enriched_df)


def show_results(enriched_df: pd.DataFrame):
    """Render charts and sample outputs for an enriched dataframe."""

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Titles", len(enriched_df))
    with col2:
        pos = (enriched_df["description_emotion_sentiment"] == "positive").sum()
        st.metric("Positive", pos)
    with col3:
        st.metric("Avg Readability", f"{enriched_df['description_readability_score'].mean():.1f}")
    with col4:
        st.metric("Avg Word Count", f"{enriched_df['nlp_word_count'].mean():.0f}")

    st.header("Emotion Distribution")
    try:
        emotion_counts = enriched_df["description_emotion_sentiment"].value_counts()
        fig = px.pie(values=emotion_counts.values, names=emotion_counts.index, title="Description Sentiment")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.write("Emotion chart unavailable")

    st.header("Sample Rows")
    sample_cols = [c for c in ["title", "type", "description_emotion_sentiment", "nlp_word_count", "nlp_summary"] if c in enriched_df.columns]
    st.dataframe(enriched_df[sample_cols].head(10), use_container_width=True)

    st.header("Single Description Analyzer")
    user_description = st.text_area("Enter a description to analyze", height=120)
    if st.button("Analyze Description") and user_description.strip():
        with st.spinner("Analyzing..."):
            emotion = analyze_emotion(user_description)
            readability = analyze_readability(user_description)
            summary = generate_ai_summary(user_description)
            words = count_words(user_description)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Emotion")
            st.write(emotion)
        with c2:
            st.subheader("Readability")
            st.write(readability)

        st.subheader("AI Summary")
        st.write(summary)
        st.subheader("Word Count")
        st.write(words)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            st.error(f"App error: {exc}")
        except Exception:
            print(f"App error: {exc}")