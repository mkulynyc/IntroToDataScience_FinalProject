'''
Import Statements
'''
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

'''
Data Loading and Cleaning
'''
### Load the Netflix data set
def load_data(file_path = "data/netflix_titles.csv"):
    df = pd.read_csv(file_path)
    return df

### Clean the Netflix dataset
### Functions to clean the data:
def cleanNetflixData(df,
                     estimateSeasonMinutes=False,
                     episodesPerSeason=10,
                     minutesPerEpisode=45,
                     explodeGenres=False,
                     standardizeGenres=True): # Added option to standardize genres

    df = df.copy()

    # --- 0) Trim whitespace in all object columns & normalize empties ---
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    # --- 1) Dates: parse and derive parts ---
    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
        df["year_added"]  = df["date_added"].dt.year.astype("Int64")
        df["month_added"] = df["date_added"].dt.month.astype("Int64")

    # --- 2) Type normalization ---
    if "type" in df.columns:
        df["type"] = df["type"].str.title()  # 'Movie' / 'Tv Show' -> 'Movie' / 'Tv Show'
        df["type"] = df["type"].replace({"Tv Show": "TV Show"})

    # --- 3) Rating normalization (unify common TV-* variants) ---
    if "rating" in df.columns:
        r = df["rating"].str.upper().str.replace(" ", "-", regex=False)
        tv_fixes = {
            "TV-MA": "TV-MA", "TV-14": "TV-14", "TV-PG": "TV-PG", "TV-G": "TV-G",
            "TV-Y7": "TV-Y7", "TV-Y": "TV-Y"
        }
        # normalize common spaced forms like "TV MA", "TV 14", etc.
        r = r.replace({
            "TVMA": "TV-MA", "TV14": "TV-14", "TVPG": "TV-PG", "TVG": "TV-G",
            "TVY7": "TV-Y7", "TVY": "TV-Y"
        })
        df["rating"] = r.replace(tv_fixes)

    # --- 4) Duration: extract minutes and seasons ---
    df["duration_minutes"] = pd.to_numeric(
        df.get("duration", pd.Series(index=df.index, dtype="object"))
          .str.extract(r"(\d+)\s*min", expand=False), errors="coerce"
    ).astype("Int64")

    df["seasons"] = pd.to_numeric(
        df.get("duration", pd.Series(index=df.index, dtype="object"))
          .str.extract(r"(\d+)\s*Season", flags=re.IGNORECASE, expand=False),
        errors="coerce"
    ).astype("Int64")

    # Optionally estimate TV show minutes
    if estimateSeasonMinutes:
        est = df["seasons"].astype("Float64") * episodesPerSeason * minutesPerEpisode
        df["duration_minutes"] = df["duration_minutes"].astype("Float64")
        df["duration_minutes"] = df["duration_minutes"].fillna(est).round().astype("Int64")

    # --- 5) Countries: split & primary country ---
    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unknown")
        df["countries"] = df["country"].str.split(r"\s*,\s*")
        df["primary_country"] = df["countries"].apply(lambda xs: xs[0] if isinstance(xs, list) and len(xs) else "Unknown")

    # --- 6) Director / Cast: fill NaN with 'Unknown' for easier grouping & add flags ---
    for c in ["director", "cast"]:
        if c in df.columns:
            df[f"has_{c}"] = df[c].notna() # Add flag for missing director/cast
            df[c] = df[c].fillna("Unknown")


    # --- 7) Genres: split; optionally explode & standardize ---
    if "listed_in" in df.columns:
        df["genres"] = df["listed_in"].fillna("Unknown").str.split(r"\s*,\s*")
        if standardizeGenres:
             # Simple standardization: remove " TV Shows", " Movies", " Series" etc.
            df["genres"] = df["genres"].apply(lambda genre_list: [
                re.sub(r'\s*(TV Shows?|Movies?|Series?|Dramas?)$', '', genre, flags=re.IGNORECASE).strip()
                for genre in genre_list
            ])
    else:
        df["genres"] = [[] for _ in range(len(df))]


    genres_exploded = None
    if explodeGenres:
        genres_exploded = (
            df[["show_id", "title", "genres"]]
            .explode("genres", ignore_index=False)
            .rename(columns={"genres": "genre"})
            .reset_index(drop=False)
            .rename(columns={"index": "row_idx"})
        )

    # --- 8) Dtypes & duplicates ---
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")
        # Basic validation for release year (e.g., not in the future, not extremely old)
        current_year = pd.Timestamp('now').year
        df.loc[df['release_year'] > current_year + 1, 'release_year'] = pd.NA # Assuming release year shouldn't be more than 1 year in the future
        df.loc[df['release_year'] < 1900, 'release_year'] = pd.NA # Assuming release year shouldn't be before 1900 (adjust as needed)


    df = df.drop_duplicates(subset=["show_id"]).reset_index(drop=True)

    # --- 9) Helpful flags ---
    df["is_movie"] = (df.get("type", "") == "Movie")
    df["is_tv"]    = (df.get("type", "") == "TV Show")


    # --- 10) Extract release month and day (if release_year is valid) ---
    if "release_year" in df.columns:
        # Create a temporary date column to extract month and day, handling NaT
        temp_date = pd.to_datetime(df['release_year'], format='%Y', errors='coerce')
        df['release_month'] = temp_date.dt.month.astype('Int64')
        df['release_day'] = temp_date.dt.day.astype('Int64')


    return (df, genres_exploded) if explodeGenres else (df, None)

### Extract only movies
def movies_only(df):
    df_movies = df[df['type'] == 'Movie'].copy()
    return df_movies

### Add user ratings
def load_imdb_cleaned(file_path="data/imdb_movies_cleaned.csv"):
    df = pd.read_csv(file_path)
    return df



'''
Movie Search/Recommender Engine based on keywords and genres
'''
### Genre Preparation
def add_genres_list(df):
    """
    Adds a genres_list column for filtering and displays unique genres in Streamlit.
    """
    df = df.copy()
    genre_lists = df['genres'].apply(
        lambda x: x if isinstance(x, list) else [g.strip() for g in str(x).split(',') if g.strip()]
    )
    df['genres_list'] = genre_lists

    unique_genres = sorted(set(g for sublist in genre_lists for g in sublist))
    st.markdown("### 🎬 Unique Genres")
    st.markdown(", ".join(unique_genres))

    return df

### Keyword matching function
def keyword_match(df, keywords, match_mode='any'):
    """
    Filters movies whose description contains any or all of the keywords.
    """
    if match_mode == 'all':
        return df[df['description'].apply(
            lambda desc: all(re.search(re.escape(k), str(desc), re.IGNORECASE) for k in keywords)
        )]
    else:
        pattern = '|'.join([re.escape(k) for k in keywords])
        return df[df['description'].str.contains(pattern, case=False, na=False)]

### Filtering to only include movies that match the genres provided
def genre_filter(df, genre_list, match_mode='any'):
    """
    Filters movies that match any or all of the genres provided.
    """
    genre_list_lower = [g.lower() for g in genre_list]

    def match_genres(g):
        g_lower = [x.lower() for x in g if x]
        if match_mode == 'all':
            return all(genre in g_lower for genre in genre_list_lower)
        else:
            return any(genre in g_lower for genre in genre_list_lower)

    return df[df['genres_list'].apply(match_genres)]

### Recommending top movies based on inputs
def recommend_movies(df, keywords=None, genres=None, top_n=10, keyword_match_mode='any', genre_match_mode='any'):
    """
    Recommends top-rated movies based on keyword and genre filters.
    Match modes: 'any' or 'all' for both keywords and genres.
    """
    filtered = df.copy()

    if keywords:
        filtered = keyword_match(filtered, keywords, match_mode=keyword_match_mode)
        print(f"🔍 Keyword filter: {len(filtered)} matches for {keywords} ({keyword_match_mode})")

    if genres:
        filtered = genre_filter(filtered, genres, match_mode=genre_match_mode)
        print(f"🎬 Genre filter: {len(filtered)} matches for {genres} ({genre_match_mode})")

    print(f"✅ Final filtered set: {len(filtered)} movies")

    if filtered.empty:
        print("⚠️ No matches found. Try relaxing your keywords or genre filters.")
        return pd.DataFrame(columns=['title', 'averageRating', 'genres', 'description'])

    return (
        filtered
        .sort_values(by='averageRating', ascending=False)
        .loc[:, ['title', 'averageRating', 'genres', 'description']]
        .head(top_n)
    )


### Run the recommender
def run_recommender(df, keywords, genres, top_n=10, keyword_match_mode='any', genre_match_mode='any'):
    """
    Wrapper for Streamlit app to run the recommender system.
    """
    return recommend_movies(
        df,
        keywords=keywords,
        genres=genres,
        top_n=top_n,
        keyword_match_mode=keyword_match_mode,
        genre_match_mode=genre_match_mode
    )
    

'''
Data Visualizations
'''

### Rating (M, PG, etc.) Per Year Plot
def plot_rating_counts_by_year(df):
    df_movies_ratings = df.dropna(subset=['release_year', 'rating'])
    df_movies_ratings = df_movies_ratings[df_movies_ratings['release_year'] >= 2016]

    rating_counts = df_movies_ratings.groupby(['release_year', 'rating']).size().unstack(fill_value=0)
    rating_order = rating_counts.sum().sort_values(ascending=False).index
    rating_counts = rating_counts[rating_order]

    fig, ax = plt.subplots(figsize=(10, 8))
    bottom = pd.Series([0]*rating_counts.shape[0], index=rating_counts.index)
    colors = plt.cm.tab20.colors

    for i, rating in enumerate(rating_counts.columns):
        ax.barh(
            rating_counts.index,
            rating_counts[rating],
            left=bottom,
            label=rating,
            color=colors[i % len(colors)]
        )
        for y, x in enumerate(bottom + rating_counts[rating]/2):
            count = rating_counts[rating].iloc[y]
            if count > 20:
                ax.text(x, rating_counts.index[y], str(count),
                        ha='center', va='center', fontsize=9, color='white')
        bottom += rating_counts[rating]

    ax.set_xlabel('Number of Movies')
    ax.set_ylabel('Release Year')
    ax.set_title('Stacked Count of Movie Ratings from 2016 Onwards (Counts > 20)')
    ax.legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

### Same but as styled table
def show_rating_table(df):
    df_movies_ratings = df.dropna(subset=['release_year', 'rating'])
    df_movies_ratings = df_movies_ratings[df_movies_ratings['release_year'] >= 2016]

    rating_counts = df_movies_ratings.groupby(['release_year', 'rating']).size().unstack(fill_value=0)
    styled_table = rating_counts.style.background_gradient(cmap='YlGnBu', axis=None)\
                                       .set_caption("Movie Ratings Count per Year")
    st.dataframe(styled_table)
    
### Plot top genres by country over time
def plot_top_genres_by_country(df, country='United States', top_n=5):
    df_country = df[df['country'] == country].dropna(subset=['listed_in', 'release_year'])
    df_genres = df_country.assign(genre=df_country['listed_in'].str.split(', ')).explode('genre')

    genre_counts = df_genres.groupby(['release_year', 'genre']).size().reset_index(name='count')
    top_genres = genre_counts.groupby('genre')['count'].sum().sort_values(ascending=False).head(top_n).index
    genre_counts_top = genre_counts[genre_counts['genre'].isin(top_genres)]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=genre_counts_top, x='release_year', y='count', hue='genre', marker='o', ax=ax)
    ax.set_title(f'Top {top_n} Genres in {country} Over Time')
    ax.set_xlabel('Release Year')
    ax.set_ylabel('Number of Projects')
    ax.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(fig)


### Actor Timeline
def plot_actor_timeline(df, actor_name):
    df_actor = df[df['cast'].str.contains(actor_name, na=False)].copy()
    df_actor = df_actor.sort_values("release_year")

    df_actor["start"] = pd.to_datetime(
        df_actor["release_year"].apply(lambda x: str(int(float(x))) if pd.notna(x) else None),
        format="%Y", errors="coerce"
    )
    df_actor = df_actor.dropna(subset=["start"])
    df_actor["end"] = df_actor["start"] + pd.DateOffset(months=6)

    fig = px.timeline(
        df_actor,
        x_start="start",
        x_end="end",
        y="title",
        color="type" if "type" in df_actor.columns else None,
        hover_data=["release_year", "cast", "genres", "averageRating"],
        color_discrete_map={"Movie": "skyblue", "TV Show": "lightgreen"}
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=f"{actor_name} Filmography Timeline",
        xaxis_title="Release Year",
        yaxis_title="Projects"
    )
    st.plotly_chart(fig, use_container_width=True)
