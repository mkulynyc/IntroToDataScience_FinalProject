# Import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import streamlit as st
from tabulate import tabulate
import plotly.express as px

# Function to load the data
def load_data(file_path = "data/netflix_titles.csv"):
    df = pd.read_csv(file_path)
    return df

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

# Get a separate movie and TV dataset
def movies_and_tv_split(df):
    return df[df['type'] == 'Movie'].copy(), df[df['type'] == 'TV Show'].copy()

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
    return df

