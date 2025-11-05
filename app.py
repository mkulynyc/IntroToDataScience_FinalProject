import streamlit as st
from apputil import *

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# Load Netflix and IMDb data
df_netflix_raw = load_data()
df_netflix_clean, _ = cleanNetflixData(df_netflix_raw)
# Load IMDb cleaned data
df_imdb = load_imdb_cleaned()

# Load and clean Netflix data
df_netflix_raw = load_data()
df_netflix_clean, _ = cleanNetflixData(df_netflix_raw)
df_netflix_movies = movies_only(df_netflix_clean)

# Merge all relevant Netflix columns into IMDb
df_imdb_enriched = df_imdb.merge(
    df_netflix_movies[[
        "title", "genres", "description", "rating", "listed_in",
        "country", "cast", "type"
    ]],
    how="left",
    left_on="primaryTitle",
    right_on="title"
)


# Drop the redundant Netflix title column
df_imdb_enriched = df_imdb_enriched.drop(columns=["title"])

# Add genres_list for filtering
df_imdb_ready = add_genres_list(df_imdb_enriched)

# Rename for consistency with plotting and recommender functions
df_imdb_ready = df_imdb_ready.rename(columns={
    "primaryTitle": "title",
    "startYear": "release_year"
})

# Setting up visual

### Sidebar for user inputs
st.sidebar.header("🔍 Filter Options")
keywords_input = st.sidebar.text_input("Keywords (comma-separated)", value="school")
selected_genres = st.sidebar.multiselect("Genres", options=sorted(set(g for sublist in df_imdb_ready['genres_list'] for g in sublist)))
keyword_mode = st.sidebar.radio("Keyword Match Mode", ["any", "all"])
genre_mode = st.sidebar.radio("Genre Match Mode", ["any", "all"])
top_n = st.sidebar.slider("Number of Recommendations", 1, 20, 10)

### Recommender output
st.header("🎯 Recommended Movies")
if st.sidebar.button("Run Recommender"):
    keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
    results = run_recommender(df_imdb_ready, keywords, selected_genres, top_n, keyword_mode, genre_mode)
    if results.empty:
        st.warning("No matches found.")
    else:
        st.dataframe(results.reset_index(drop=True), use_container_width=True)
        

### Visualizations
st.header("📊 Ratings Over Time")
plot_rating_counts_by_year(df_imdb_ready)
show_rating_table(df_imdb_ready)

st.header("🌍 Top Genres by Country")
country = st.text_input("Country", value="United States")
plot_top_genres_by_country(df_imdb_ready, country)

st.header("🎬 Actor Timeline")
actor_name = st.text_input("Actor Name", value="Liam Neeson")
if actor_name:
    plot_actor_timeline(df_imdb_ready, actor_name)