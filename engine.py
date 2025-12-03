from data_load import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import streamlit as st
from tabulate import tabulate
import plotly.express as px
from rapidfuzz import fuzz

### Keyword fuzzy matching function
def keyword_match_fuzzy(df, keywords, match_mode='any', threshold=70):
    """
    Filters movies whose description contains fuzzy matches of the keywords.
    - threshold: similarity score (0–100) for fuzzy matching
    """
    def match_desc(desc):
        tokens = str(desc).lower().split()
        if match_mode == 'all':
            return all(
                any(fuzz.partial_ratio(k.lower(), token) >= threshold for token in tokens)
                for k in keywords
            )
        else:
            return any(
                any(fuzz.partial_ratio(k.lower(), token) >= threshold for token in tokens)
                for k in keywords
            )
    return df[df['description'].apply(match_desc)]

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
        filtered = keyword_match_fuzzy(filtered, keywords, match_mode=keyword_match_mode, threshold=fuzzy_threshold)
        print(f"🔍 Keyword filter: {len(filtered)} matches for {keywords} ({keyword_match_mode}, threshold={fuzzy_threshold})")

    if genres:
        filtered = genre_filter(filtered, genres, match_mode=genre_match_mode)
        print(f"🎬 Genre filter: {len(filtered)} matches for {genres} ({genre_match_mode})")

    print(f"✅ Final filtered set: {len(filtered)} movies")

    if filtered.empty:
        print("⚠️ No matches found. Try relaxing your keywords or genre filters.")
        return pd.DataFrame(columns=['title', 'rating', 'genres', 'description'])

    return (
        filtered
        .sort_values(by='title', ascending=False)
        .loc[:, ['title', 'rating', 'genres', 'description']]
        .head(top_n)
    )


### Run the recommender
def run_recommender(df, keywords, genres, top_n=10, keyword_match_mode='any', genre_match_mode='any', fuzzy_threshold=70):
    """
    Wrapper for Streamlit app to run the recommender system.
    """
    return recommend_movies(
        df,
        keywords=keywords,
        genres=genres,
        top_n=top_n,
        keyword_match_mode=keyword_match_mode,
        genre_match_mode=genre_match_mode,
        fuzzy_threshold=fuzzy_threshold
    )