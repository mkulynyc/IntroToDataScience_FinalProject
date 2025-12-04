import os
import time
from typing import List, Optional

import pandas as pd
import requests
import streamlit as st

#from nlp_utils import scoreDataFrame
from nlp.spacy_model import evaluateSpacy
#from viz import plotVaderVsSpacy, plotLabelCounts

from plots import *
from engine import *
from data_load import *
from src.utils.shared_utils import analyze_emotion, analyze_readability, count_words

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

tabs = st.tabs([
    "📊 Overview",
    "🔎 Explore",
    "⚖️ Model Compare",
    "🧭 Title Explorer",
    "⚙️ Ingest & Score",
    "🎯 Recommender Engine",
    "📊 Visualizations", 
     "🧠 Text Enrichment"
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
        
        ####### COMMMENTING OUT TO RUN ########
        '''
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
        '''
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
        
        ######## COMMENTING OUT TO RUN
        
        '''
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
        '''
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

        ### COMMENTING OUT TO RUN
        
        '''
        st.markdown("#### VADER vs spaCy")
        try:
            scatter = plotVaderVsSpacy(df, textCol="nlp_text")
            st.plotly_chart(scatter, use_container_width=True)
        except Exception as e:
            st.warning(f"Chart error: {e}")
        '''

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

    from fetch_tmdb_reviews_test import main as fetch_reviews
    from enrich_and_score_test import main as enrich_and_score

    c1, c2 = st.columns(2)

    with c1:
        if st.button("🔄 Fetch TMDB Reviews now", use_container_width=True):
            try:
                fetch_reviews(netflix_path=NETFLIX_PATH, output_path=REVIEWS_PATH, limit=300)
                st.success("Reviews fetched and saved!")
            except Exception as e:
                st.error(f"Failed to fetch reviews: {e}")

    with c2:
        if st.button("⚙️ Enrich + Score now", use_container_width=True):
            try:
                enrich_and_score(
                    netflix_path=NETFLIX_PATH,
                    reviews_path=REVIEWS_PATH,
                    output_path=ENRICHED_PATH,
                    spacy_model_path=SPACY_MODEL_PATH
                )
                st.success("Enrichment complete!")
            except Exception as e:
                st.error(f"Enrichment failed: {e}")

    st.markdown("---")
    st.caption("Need to just score a small CSV on the fly? Upload it below (uses your trained spaCy model).")
    up = st.file_uploader("Upload CSV with a 'text' column", type=["csv"])

    # Uncomment when ready
    '''
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
    '''
    
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
    
# ---------- Text Enrichment ----------
with tabs[7]:
    st.header("🧠 Text Enrichment Workbench")

    df_raw = loadCsv(NETFLIX_PATH)
    df_clean, _ = cleanNetflixData(df_raw)
    df_clean = add_genres_list(df_clean)

    if df_clean is None or df_clean.empty:
        st.info("No Netflix data found.")
    else:
        st.subheader("Select a Movie or TV Show")

        # Dropdown with autocomplete
        title = st.selectbox(
            "Choose a title",
            options=sorted(df_clean["title"].dropna().unique()),
            index=0,
            placeholder="Start typing a title..."
        )

        # Get the row for the selected title
        row = df_clean[df_clean["title"] == title].iloc[0]
        description = row.get("description", "")

        st.markdown("### Description")
        st.text_area("Description", value=description, height=160)

        if description:
            # Run enrichment functions
            emotion = analyze_emotion(description)
            readability = analyze_readability(description)
            word_count = count_words(description)

            st.markdown("### Analysis Results")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Sentiment", emotion['sentiment'])
                st.caption(f"Polarity: {emotion['polarity']:.2f}, Subjectivity: {emotion['subjectivity']:.2f}")

            with c2:
                st.metric("Readability", readability['category'])
                st.caption(f"Flesch Score: {readability['score']:.1f}")

            with c3:
                st.metric("Word Count", word_count)