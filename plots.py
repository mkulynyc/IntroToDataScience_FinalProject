# Import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import streamlit as st
from tabulate import tabulate
import plotly.express as px
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

### Styled table of movie/tv show ratings
def show_rating_table(df, year=2016, label="Movies and TV Shows"):
    df_ratings = df.dropna(subset=['release_year', 'rating'])
    df_ratings = df_ratings[df_ratings['release_year'] >= year]

    rating_counts = df_ratings.groupby(['release_year', 'rating']).size().unstack(fill_value=0)
    styled_table = rating_counts.style.background_gradient(cmap='YlGnBu', axis=None)\
                                       .set_caption(f"{label} Ratings Count per Year (Since {year})")
    st.dataframe(styled_table)
    
    
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
    
### Sentiment by Rating
def plot_sentiment_by_rating(df):
    maturity_order = [
        "TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "G", "PG", "TV-PG",
        "PG-13", "TV-14", "R", "TV-MA", "NC-17", "UR", "NR",
        "66-MIN", "74-MIN", "84-MIN"
    ]
    
    # Analyze
    analyzer = SentimentIntensityAnalyzer()

    # Apply to your NLP text column (e.g., description or nlp_text)
    df["vader_compound"] = df["description"].fillna("").astype(str).map(
        lambda text: analyzer.polarity_scores(text)["compound"]
    )
    
    df_sent = df[df["vader_compound"].notna() & df["rating"].notna()]
    df_sent = df_sent[df_sent["rating"].isin(maturity_order)]

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_sent, x="rating", y="vader_compound", order=maturity_order, palette="coolwarm")
    plt.title("Sentiment Score Distribution by Rating (Sorted by Maturity)")
    plt.xlabel("Rating")
    plt.ylabel("VADER Compound Sentiment Score")
    plt.xticks(rotation=45)
    plt.figtext(0.5, -0.05,
                "Boxplots show sentiment score distributions for each rating, ordered by increasing maturity.",
                wrap=True, horizontalalignment='center', fontsize=9)
    plt.tight_layout()
    plt.show()
    
### Top Actors Plot::
def plot_top_actors(df, top_n=20):
    # Flatten and count actors, excluding "Unknown"
    all_actors = df["cast"].dropna().str.split(", ").explode()
    filtered_actors = all_actors[all_actors.str.lower() != "unknown"]
    actor_counts = Counter(filtered_actors)

    # Top 20
    top_actors = pd.DataFrame(actor_counts.most_common(20), columns=["actor", "count"])

    # Plot
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_actors, y="actor", x="count", palette="viridis")
    plt.title("Top 20 Most Frequent Actors in Netflix Titles (Excluding 'Unknown')")
    plt.xlabel("Number of Appearances")
    plt.ylabel("Actor")
    plt.tight_layout()
    plt.show()
    
### Volume by Country plot
def plot_country_distribution(df):
    # Count titles per country
    country_counts = df["primary_country"].value_counts().reset_index()
    country_counts.columns = ["country", "count"]

    # Plot
    fig = px.choropleth(country_counts,
                        locations="country",
                        locationmode="country names",
                        color="count",
                        color_continuous_scale="Reds",
                        title="Netflix Content Volume by Country")
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    fig.show()
    
    
### Movie vs Tv over time plot
def plot_movie_tv_trends(df):
    # Filter valid years
    df_years = df[df["release_year"].notna() & df["type"].notna()]

    # Group and count
    counts = df_years.groupby(["release_year", "type"]).size().reset_index(name="count")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=counts, x="release_year", y="count", hue="type", marker="o")
    plt.title("Movie vs TV Show Volume Over Time")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Titles")
    plt.figtext(0.5, -0.05,
                "Each line shows the number of Netflix Movies and TV Shows released per year.",
                wrap=True, horizontalalignment='center', fontsize=9)
    plt.tight_layout()
    plt.show()