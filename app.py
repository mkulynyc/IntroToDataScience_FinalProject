import os
import re
import time
from typing import List, Optional

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import textstat

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
# Data processing functions
# =========================
def cleanNetflixData(df):
    """Clean and prepare Netflix data."""
    if df is None:
        return None, "No data provided"
    df_clean = df.copy()
    # Basic cleaning
    df_clean = df_clean.dropna(subset=['title'])
    if 'date_added' in df_clean.columns:
        df_clean['date_added'] = pd.to_datetime(df_clean['date_added'], errors='coerce')
    if 'release_year' in df_clean.columns:
        df_clean['release_year'] = pd.to_numeric(df_clean['release_year'], errors='coerce')
    return df_clean, None

def add_genres_list(df):
    """Add a genres_list column from listed_in."""
    if df is None:
        return df
    if "listed_in" in df.columns:
        df["genres_list"] = df["listed_in"].fillna("").astype(str).apply(
            lambda s: [g.strip() for g in s.split(",")] if s else []
        )
    else:
        df["genres_list"] = [[] for _ in range(len(df))]
    return df

def run_recommender(df, keywords, genres, top_n=10, keyword_mode='any', genre_mode='any'):
    """Simple recommender based on keywords and genres."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    results = df.copy()
    
    # Filter by keywords in description
    if keywords and 'description' in df.columns:
        def match_keywords(desc):
            if pd.isna(desc):
                return False
            desc_lower = str(desc).lower()
            matches = [kw.lower() in desc_lower for kw in keywords]
            return all(matches) if keyword_mode == 'all' else any(matches)
        
        results = results[results['description'].apply(match_keywords)]
    
    # Filter by genres
    if genres and 'genres_list' in df.columns:
        def match_genres(genre_list):
            if not genre_list:
                return False
            matches = [g in genre_list for g in genres]
            return all(matches) if genre_mode == 'all' else any(matches)
        
        results = results[results['genres_list'].apply(match_genres)]
    
    # Return top N
    display_cols = ['title', 'type', 'release_year', 'rating', 'listed_in', 'description']
    available_cols = [col for col in display_cols if col in results.columns]
    return results[available_cols].head(top_n)

def show_rating_table(df, year=2016):
    """Display a table of ratings grouped by year."""
    if df is None or df.empty:
        st.warning("No data available")
        return
    
    if 'rating' not in df.columns or 'release_year' not in df.columns:
        st.warning("Missing required columns for ratings table")
        return
    
    filtered = df[df['release_year'] >= year].copy()
    if filtered.empty:
        st.warning(f"No data for year >= {year}")
        return
    
    rating_counts = filtered.groupby(['release_year', 'rating']).size().reset_index(name='count')
    pivot_table = rating_counts.pivot(index='release_year', columns='rating', values='count').fillna(0)
    st.dataframe(pivot_table, use_container_width=True)

def plot_top_genres_by_country(df, country='United States', top_n=5):
    """Plot top genres for a specific country over time."""
    if df is None or df.empty:
        st.warning("No data available")
        return
    
    if 'country' not in df.columns or 'genres_list' not in df.columns:
        st.warning("Missing required columns for genre analysis")
        return
    
    # Filter by country
    country_data = df[df['country'].fillna('').str.contains(country, case=False, na=False)].copy()
    
    if country_data.empty:
        st.warning(f"No data found for {country}")
        return
    
    # Count genres
    from collections import Counter
    all_genres = []
    for genres in country_data['genres_list']:
        all_genres.extend(genres)
    
    genre_counts = Counter(all_genres)
    top_genres = [g for g, _ in genre_counts.most_common(top_n)]
    
    st.write(f"Top {top_n} genres in {country}:")
    for i, (genre, count) in enumerate(genre_counts.most_common(top_n), 1):
        st.write(f"{i}. {genre}: {count} titles")

def plotVaderVsSpacy(df, textCol='nlp_text'):
    """Create a scatter plot comparing VADER and spaCy sentiment."""
    if 'vader_compound' not in df.columns or 'spacy_pos_prob' not in df.columns:
        return go.Figure().add_annotation(text="Missing required columns", showarrow=False)
    
    fig = px.scatter(
        df,
        x='vader_compound',
        y='spacy_pos_prob',
        color='spacy_label',
        hover_data=['title'] if 'title' in df.columns else None,
        title='VADER vs spaCy Sentiment',
        labels={'vader_compound': 'VADER Compound', 'spacy_pos_prob': 'spaCy Positive Prob'}
    )
    return fig

def plotLabelCounts(df, which='spacy_label'):
    """Create a bar chart of label counts."""
    if which not in df.columns:
        return go.Figure().add_annotation(text=f"Column {which} not found", showarrow=False)
    
    counts = df[which].value_counts()
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        title=f'{which} Distribution',
        labels={'x': 'Label', 'y': 'Count'}
    )
    return fig

# =========================
# Sidebar (pipeline actions)
# =========================
with st.sidebar:
    st.header("Pipeline")
    st.caption("Fetch → Enrich/Score → Explore")
    fetch_btn = st.button("🔄 Fetch TMDB Reviews (append)", use_container_width=True)
    score_btn = st.button("⚙️ Enrich + Score (VADER + spaCy)", use_container_width=True)

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
    st.caption("Need to just score a small CSV on the fly? Upload it below.")
    up = st.file_uploader("Upload CSV with a 'text' column", type=["csv"])
    if up:
        try:
            raw = pd.read_csv(up)
            if 'text' not in raw.columns:
                st.error("CSV must have a 'text' column")
            else:
                # Simple sentiment scoring
                raw['sentiment'] = raw['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0)
                raw['sentiment_label'] = raw['sentiment'].apply(lambda x: 'POSITIVE' if x > 0.1 else ('NEGATIVE' if x < -0.1 else 'NEUTRAL'))
                st.success("Scored! Preview below.")
                st.dataframe(raw.head(100))
                st.download_button("Download scored CSV",
                                   data=raw.to_csv(index=False).encode("utf-8"),
                                   file_name="scored.csv",
                                   use_container_width=True)
        except Exception as e:
            st.error(f"Scoring failed: {e}")

# ---------- Recommender Search Engine ---------- 
with tabs[5]:
    # Load data
    df_raw = loadCsv(NETFLIX_PATH)
    if df_raw is not None:
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
            # Safe genre extraction
            genre_options = []
            if 'genres_list' in df_clean.columns:
                for sublist in df_clean['genres_list']:
                    if sublist:
                        genre_options.extend([g for g in sublist if g])
            
            selected_genres = st.multiselect(
                "Genres",
                options=sorted(set(genre_options)) if genre_options else []
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
    else:
        st.info("Please load data/netflix_titles.csv first")
            
# ---------- Visualizations ---------
with tabs[6]:
    # Load data
    df_raw = loadCsv(NETFLIX_PATH)
    if df_raw is not None:
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
    else:
        st.info("Please load data/netflix_titles.csv first")

# ---------- Description Analyzer ----------
with tabs[7]:

    st.header("📝 Simple Description Analyzer")
    st.write("Upload a CSV containing a column named **description** to analyze text sentiment, readability, and keywords.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            if "description" not in df.columns:
                st.error("Your CSV must contain a column named 'description'.")
            else:
                st.success("File uploaded. Running analysis...")

                # --- Basic sentiment analysis (TextBlob) ---
                def get_sentiment(text):
                    try:
                        return TextBlob(str(text)).sentiment.polarity
                    except:
                        return 0
                df["sentiment"] = df["description"].apply(get_sentiment)
                # Sentiment label
                df["sentiment_label"] = df["sentiment"].apply(
                    lambda x: "Positive" if x > 0.1 else ("Negative" if x < -0.1 else "Neutral")
                )
                # --- Readability (textstat) ---
                df["readability_score"] = df["description"].apply(lambda t: textstat.flesch_reading_ease(str(t)))
                # --- Simple keyword extraction ---
                def extract_keywords(text):
                    words = re.findall(r"\b\w+\b", str(text).lower())
                    stopwords = {"the","and","a","to","of","in","is","on","it","for","with","at","from"}
                    keywords = [w for w in words if w not in stopwords]
                    return ", ".join(keywords[:5])  # top 5 words
                df["keywords"] = df["description"].apply(extract_keywords)
                # Show preview
                st.subheader("Preview of Results")
                st.dataframe(df.head(50), use_container_width=True)

                st.subheader("📊 Visualization Insights")
                # ---- Sentiment Pie Chart ----
                st.write("### Sentiment Distribution")
                sent_counts = df["sentiment_label"].value_counts()
                fig_sent = px.pie(
                    names=sent_counts.index,
                    values=sent_counts.values,
                    title="Sentiment Breakdown",
                    color=sent_counts.index,
                    color_discrete_map={
                        "Positive": "#16a34a",
                        "Neutral": "#6b7280",
                        "Negative": "#dc2626"
                    }
                )
                st.plotly_chart(fig_sent, use_container_width=True)
                # ---- Readability Histogram ----
                st.write("### Readability Score Distribution")
                fig_read = px.histogram(
                    df,
                    x="readability_score",
                    nbins=20,
                    title="Distribution of Readability (Flesch Score)"
                )
                st.plotly_chart(fig_read, use_container_width=True)
                # ---- Keyword Frequency Chart ----
                st.write("### Top Keywords")
                all_keywords = []
                for kw_list in df["keywords"]:
                    for w in kw_list.split(", "):
                        if len(w.strip()) > 1:
                            all_keywords.append(w.strip())
                from collections import Counter
                kw_counts = Counter(all_keywords).most_common(15)
                kw_df = pd.DataFrame(kw_counts, columns=["keyword", "count"])
                fig_kw = px.bar(
                    kw_df,
                    x="keyword",
                    y="count",
                    title="Top Keywords Found in Descriptions"
                )
                st.plotly_chart(fig_kw, use_container_width=True)
                # Download
                st.download_button(
                    label="Download Analyzed CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="description_analysis.csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error reading file: {e}")

