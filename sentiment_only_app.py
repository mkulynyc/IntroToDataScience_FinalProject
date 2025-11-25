import os
from typing import Optional

import pandas as pd
import streamlit as st
import plotly.express as px

# Minimal sentiment-only Streamlit app so reviewers can open a focused PR.
st.set_page_config(page_title="Sentiment Only — Netflix", layout="wide")

ENRICHED_PATH = "data/netflix_enriched_scored.csv"

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
    # derive genre list if present
    if "listed_in" in df.columns:
        df["genres_list"] = df["listed_in"].fillna("").astype(str).apply(
            lambda s: [g.strip() for g in s.split(",")] if s else []
        )
    else:
        df["genres_list"] = [[] for _ in range(len(df))]
    return df

# --- App UI ---
st.title("🎯 Sentiment Analysis (VADER + spaCy) — Focused View")

df = loadEnriched(ENRICHED_PATH)
if df is None or df.empty:
    st.info("No enriched dataset found at `data/netflix_enriched_scored.csv`. Use the main pipeline to create one.")
else:
    sub = st.tabs(["🏆 Top Titles", "📊 Genre Sentiment Summary"]) 

    # ======= TOP TITLES =======
    with sub[0]:
        st.header("🏆 Top Positive & Negative Titles")
        col1, col2, col3 = st.columns([2, 2, 1])

        # Filters
        with col1:
            genres_opts = sorted({g for lst in df.get("genres_list", []) for g in lst}) if "genres_list" in df.columns else []
            selected_genres = st.multiselect("Select genres", genres_opts)
        with col2:
            type_opts = ["(all)"] + sorted(df["type"].dropna().unique()) if "type" in df.columns else ["(all)"]
            selected_type = st.selectbox("Type filter", type_opts)
        with col3:
            max_titles = int(df['title'].nunique()) if 'title' in df.columns else len(df)
            top_n = st.slider("Top N titles", 1, max(1, max_titles), min(10, max(1, max_titles)))

        s = df.copy()
        s["spacy_pos_prob"] = s.get("spacy_pos_prob", pd.Series([0.0] * len(s))).fillna(0.0)
        s["vader_compound"] = s.get("vader_compound", pd.Series([0.0] * len(s))).fillna(0.0)
        s["vader_norm"] = (s["vader_compound"] + 1.0) / 2.0
        s["combined_score"] = s["spacy_pos_prob"] + s["vader_norm"]

        # Apply filters
        if selected_genres and "genres_list" in s.columns:
            s = s[s["genres_list"].apply(lambda lst: any(g in lst for g in selected_genres))]
        if selected_type != "(all)":
            s = s[s["type"] == selected_type]

        # Drop duplicates preferring most-reviewed when available
        if "review_join" in s.columns:
            s["review_count"] = s["review_join"].fillna("").astype(str).apply(lambda x: 0 if not x else x.count(" || ") + 1)
            s = s.sort_values("review_count", ascending=False).drop_duplicates(subset=["title"], keep="first")
        else:
            s = s.drop_duplicates(subset=["title"], keep="first")

        # Positive / Negative
        pos = s.sort_values("combined_score", ascending=False).head(top_n)
        neg = s.sort_values("combined_score", ascending=True).head(top_n)

        # Helper to truncate titles in the chart while keeping full title in hover
        def truncate_title(title, length=30):
            return title if len(title) <= length else title[:length] + "..."

        pos["title_trunc"] = pos["title"].apply(truncate_title)
        figp = px.bar(
            pos.sort_values("combined_score", ascending=True),
            x="combined_score",
            y="title_trunc",
            orientation="h",
            color="combined_score",
            color_continuous_scale="Greens",
            labels={"combined_score": "Score", "title_trunc": "Title"},
            hover_data={"title": True, "type": True, "release_year": True, "spacy_pos_prob": True, "vader_compound": True},
        )
        figp.update_layout(yaxis={"categoryorder": "total ascending"})

        neg["title_trunc"] = neg["title"].apply(truncate_title)
        fign = px.bar(
            neg.sort_values("combined_score", ascending=True),
            x="combined_score",
            y="title_trunc",
            orientation="h",
            color="combined_score",
            color_continuous_scale="Reds",
            labels={"combined_score": "Score", "title_trunc": "Title"},
            hover_data={"title": True, "type": True, "release_year": True, "spacy_pos_prob": True, "vader_compound": True},
        )
        fign.update_layout(yaxis={"categoryorder": "total ascending"})

        # Display
        left_col, right_col = st.columns(2)
        with left_col:
            st.subheader("Top Positive Titles")
            st.plotly_chart(figp, width='stretch')
            st.dataframe(pos[["title", "type", "release_year", "spacy_pos_prob", "vader_compound", "combined_score"]])
            st.download_button("Download Positive CSV", data=pos.to_csv(index=False).encode("utf-8"), file_name="top_positive_titles.csv")
        with right_col:
            st.subheader("Top Negative Titles")
            st.plotly_chart(fign, width='stretch')
            st.dataframe(neg[["title", "type", "release_year", "spacy_pos_prob", "vader_compound", "combined_score"]])
            st.download_button("Download Negative CSV", data=neg.to_csv(index=False).encode("utf-8"), file_name="top_negative_titles.csv")

    # ======= GENRE SENTIMENT SUMMARY =======
    with sub[1]:
        st.header("📊 Sentiment Summary by Genre")

        type_opts = ["(all)"] + sorted(df["type"].dropna().unique()) if "type" in df.columns else ["(all)"]
        selected_type = st.selectbox("Filter by Type", type_opts)

        s2 = df.copy()
        if selected_type != "(all)":
            s2 = s2[s2["type"] == selected_type]

        if "genres_list" not in s2.columns:
            st.warning("No genre information found in dataset.")
        else:
            s2 = s2.explode("genres_list")
            s2["avg_sentiment"] = s2["spacy_pos_prob"].fillna(0) + ((s2["vader_compound"].fillna(0) + 1) / 2)
            genre_summary = (
                s2.groupby("genres_list")
                .agg(avg_sentiment=("avg_sentiment", "mean"), num_titles=("title", "count"))
                .reset_index()
                .sort_values("num_titles", ascending=False)
            )

            def sentiment_label(score):
                if score > 0.55:
                    return "Positive"
                elif score < 0.45:
                    return "Negative"
                else:
                    return "Neutral"

            genre_summary["Sentiment"] = genre_summary["avg_sentiment"].apply(sentiment_label)

            top_n_g = st.slider("Top N genres to show", 1, max(1, len(genre_summary)), 10)
            shown = genre_summary.head(top_n_g)

            st.dataframe(shown)

            fig = px.bar(
                shown,
                x="avg_sentiment",
                y="genres_list",
                color="Sentiment",
                color_discrete_map={"Positive": "#2ca02c", "Neutral": "#ffbb78", "Negative": "#d62728"},
                orientation="h",
                labels={"avg_sentiment": "Average Sentiment", "genres_list": "Genre"},
                title="Average Sentiment by Genre",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width='stretch')
