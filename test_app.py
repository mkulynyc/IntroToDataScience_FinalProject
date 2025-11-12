### Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import streamlit as st
from tabulate import tabulate
import plotly.express as px

from data_load import *
from plots import *
from engine import *

# ─────────────────────────────────────────────────────────────
# 🎬 App Config
st.set_page_config(page_title="Netflix Explorer", layout="wide")

# ─────────────────────────────────────────────────────────────
# 📦 Load and Clean Data
df_raw = load_data()
df_clean, _ = cleanNetflixData(df_raw)
df_clean = add_genres_list(df_clean)

# ─────────────────────────────────────────────────────────────
# 🧭 Tabs
tab1, tab2 = st.tabs(["🎯 Recommender Engine", "📊 Visualizations"])

# ─────────────────────────────────────────────────────────────
# 🎯 TAB 1: Recommender Engine
with tab1:
    st.header("🎯 Movie & TV Show Recommender")

    # Sidebar filters
    st.sidebar.header("🔍 Filter Options")
    keywords_input = st.sidebar.text_input("Keywords (comma-separated)", value="school")
    selected_genres = st.sidebar.multiselect(
        "Genres",
        options=sorted(set(g for sublist in df_clean['genres_list'] for g in sublist if g))
    )
    keyword_mode = st.sidebar.radio("Keyword Match Mode", ["any", "all"])
    genre_mode = st.sidebar.radio("Genre Match Mode", ["any", "all"])
    top_n = st.sidebar.slider("Number of Recommendations", 1, 20, 10)

    # Run recommender
    st.subheader("Recommended Titles")
    if st.sidebar.button("Run Recommender"):
        keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
        results = run_recommender(df_clean, keywords, selected_genres, top_n, keyword_mode, genre_mode)
        if results.empty:
            st.warning("No matches found. Try different keywords or genres.")
        else:
            st.dataframe(results.reset_index(drop=True), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 📊 TAB 2: Visualizations
with tab2:
    st.header("📊 Netflix Content Visualizations")

    # Ratings table
    st.subheader("🎬 Ratings Table (Since 2016)")
    show_rating_table(df_clean)

    # Top genres by country
    st.subheader("🌍 Top Genres by Country Over Time")
    country = st.text_input("Enter a country", value="United States")
    top_n = st.slider("Top N Genres", 3, 10, 5)
    plot_top_genres_by_country(df_clean, country=country, top_n=top_n)