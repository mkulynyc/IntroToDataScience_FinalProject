import os
import time
from typing import List, Optional

import pandas as pd
import requests
import streamlit as st

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
SPACY_MODEL_PATH = "nlp/spacy_model/artifacts/best"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185"

# =========================
# Helpers (cached)
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
    else:
        df["genres_list"] = [[] for _ in range(len(df))]
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

# Temporary visual indicator to confirm updated code is running
st.sidebar.markdown("**DEV:** top-level tab `🏆 Top Positive & Negative Titles` added")

tabs = st.tabs([
    "📊 Overview",
    "🔎 Explore",
    "⚖️ Model Compare",
    "🧭 Title Explorer",
    "⚙️ Ingest & Score",
    "🎯 Recommender Engine",
    "📊 Visualizations",
    "📈 Sentiment Summary by Genre",
    "🏆 Top Positive & Negative Titles",
    "💡 Insights"
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
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Chart error: {e}")
        with right:
            try:
                scatter = plotVaderVsSpacy(df, textCol="nlp_text")
                st.plotly_chart(scatter, use_container_width=True)
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
                st.plotly_chart(fig1, use_container_width=True)
            except Exception as e:
                st.warning(f"Chart error: {e}")
        with colB:
            try:
                fig2 = plotLabelCounts(view, which="spacy_label")
                st.plotly_chart(fig2, use_container_width=True)
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
            st.plotly_chart(scatter, use_container_width=True)
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
    top_n = st.slider("Top N Genres", 3, 10, 5)
    plot_top_genres_by_country(df_clean, country=country, top_n=top_n)
    
    
    
    
    

# ---------- Sentiment Analysis (Merged Tab) ----------
with tabs[7]:
    df = loadEnriched(ENRICHED_PATH)
    st.header("🧠 Sentiment Analysis Dashboard")

    if df is None or df.empty:
        st.info("No enriched dataset yet. Use the **Ingest & Score** tab to create it.")
    else:
        sub = st.tabs(["🏆 Top Titles", "📊 Genre Sentiment Summary"])

        # ======= TOP TITLES =======
        with sub[0]:
            st.header("🏆 Top Positive & Negative Titles")
            col1, col2, col3 = st.columns([2,2,1])

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
            s["spacy_pos_prob"] = s.get("spacy_pos_prob", pd.Series([0.0]*len(s))).fillna(0.0)
            s["vader_compound"] = s.get("vader_compound", pd.Series([0.0]*len(s))).fillna(0.0)
            s["vader_norm"] = (s["vader_compound"] + 1.0)/2.0
            s["combined_score"] = s["spacy_pos_prob"] + s["vader_norm"]

            # Apply filters
            if selected_genres and "genres_list" in s.columns:
                s = s[s["genres_list"].apply(lambda lst: any(g in lst for g in selected_genres))]
            if selected_type != "(all)":
                s = s[s["type"] == selected_type]

            # Drop duplicates
            if "review_join" in s.columns:
                s["review_count"] = s["review_join"].fillna("").astype(str).apply(lambda x: 0 if not x else x.count(" || ")+1)
                s = s.sort_values("review_count", ascending=False).drop_duplicates(subset=["title"], keep="first")
            else:
                s = s.drop_duplicates(subset=["title"], keep="first")

            # Positive / Negative
            pos = s.sort_values("combined_score", ascending=False).head(top_n)
            neg = s.sort_values("combined_score", ascending=True).head(top_n)

            import plotly.express as px

            # Truncate long titles for y-axis but keep full title in hover
            def truncate_title(title, length=30):
                return title if len(title) <= length else title[:length] + "..."

            # Positive Titles Chart
            pos["title_trunc"] = pos["title"].apply(truncate_title)
            figp = px.bar(
                pos.sort_values("combined_score", ascending=True),
                x="combined_score",
                y="title_trunc",
                orientation="h",
                color="combined_score",
                color_continuous_scale="Greens",
                labels={"combined_score":"Score","title_trunc":"Title"},
                hover_data={"title":True, "type":True, "release_year":True, "spacy_pos_prob":True, "vader_compound":True}
            )
            figp.update_layout(yaxis={"categoryorder":"total ascending"})

            # Negative Titles Chart
            neg["title_trunc"] = neg["title"].apply(truncate_title)
            fign = px.bar(
                neg.sort_values("combined_score", ascending=True),
                x="combined_score",
                y="title_trunc",
                orientation="h",
                color="combined_score",
                color_continuous_scale="Reds",
                labels={"combined_score":"Score","title_trunc":"Title"},
                hover_data={"title":True, "type":True, "release_year":True, "spacy_pos_prob":True, "vader_compound":True}
            )
            fign.update_layout(yaxis={"categoryorder":"total ascending"})

            # Display charts and tables
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("Top Positive Titles")
                st.plotly_chart(figp, use_container_width=True)
                st.dataframe(pos[["title","type","release_year","spacy_pos_prob","vader_compound","combined_score"]], use_container_width=True)
                st.download_button("Download Positive CSV", data=pos.to_csv(index=False).encode("utf-8"), file_name="top_positive_titles.csv")

            with col_right:
                st.subheader("Top Negative Titles")
                st.plotly_chart(fign, use_container_width=True)
                st.dataframe(neg[["title","type","release_year","spacy_pos_prob","vader_compound","combined_score"]], use_container_width=True)
                st.download_button("Download Negative CSV", data=neg.to_csv(index=False).encode("utf-8"), file_name="top_negative_titles.csv")

        # ======= GENRE SENTIMENT SUMMARY (interactive variant) =======
        with sub[1]:
            st.header("📊 Sentiment Summary by Genre")

            # Type filter
            type_opts = ["(all)"] + sorted(df["type"].dropna().unique()) if "type" in df.columns else ["(all)"]
            selected_type = st.selectbox("Filter by Type", type_opts)

            # Filter dataframe
            s2 = df.copy()
            if selected_type != "(all)":
                s2 = s2[s2["type"] == selected_type]

            if "genres_list" not in s2.columns:
                st.warning("No genre information found in dataset.")
            else:
                s2 = s2.explode("genres_list")

                # Compute average sentiment
                s2["avg_sentiment"] = s2["spacy_pos_prob"].fillna(0) + ((s2["vader_compound"].fillna(0) + 1) / 2)
                genre_summary = (
                    s2.groupby("genres_list")
                    .agg(avg_sentiment=("avg_sentiment", "mean"), num_titles=("title", "count"))
                    .reset_index()
                    .sort_values("num_titles", ascending=False)
                )

                # Add positive/negative/neutral label
                def sentiment_label(score):
                    if score > 0.55:
                        return "Positive"
                    elif score < 0.45:
                        return "Negative"
                    else:
                        return "Neutral"

                genre_summary["Sentiment"] = genre_summary["avg_sentiment"].apply(sentiment_label)

                # Top N genres slider
                top_n = st.slider("Top N genres to show", 1, len(genre_summary), 10)
                genre_summary = genre_summary.head(top_n)

                # Show table
                st.dataframe(genre_summary, use_container_width=True)

                # Interactive diverging bar chart
                import plotly.express as px

                fig = px.bar(
                    genre_summary,
                    x="avg_sentiment",
                    y="genres_list",
                    color="Sentiment",
                    color_discrete_map={"Positive": "#2ca02c", "Neutral": "#ffbb78", "Negative": "#d62728"},
                    orientation="h",
                    labels={"avg_sentiment": "Average Sentiment", "genres_list": "Genre"},
                    title="Average Sentiment by Genre",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)





