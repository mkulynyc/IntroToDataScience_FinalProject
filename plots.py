# Import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import streamlit as st
from tabulate import tabulate
import plotly.express as px

### Styled table of movie/tv show ratings
def show_rating_table(df, year = 2016, label = "Movies and TV Shows"):
    df_ratings = df.dropna(subset=['release_year', 'rating'])
    df_ratings = df_ratings[df_ratings['release_year'] >= year]

    rating_counts = df_ratings.groupby(['release_year', 'rating']).size().unstack(fill_value=0)
    styled_table = rating_counts.style.background_gradient(cmap='YlGnBu', axis=None)\
                                       .set_caption(f"{label} Ratings Count per Year")                         
    st.dataframe(styled_table)
    
    
### Plotting top n genres by a user-defined country
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