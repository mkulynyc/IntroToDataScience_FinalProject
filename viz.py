import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plotContentByReleaseYear(df: pd.DataFrame):
    data = df.dropna(subset=["release_year"]).groupby("release_year").size().reset_index(name="count")
    fig = px.line(data, x="release_year", y="count", title="Titles by Release Year")
    fig.update_layout(xaxis_title="Release Year", yaxis_title="Count")
    return fig

def plotContentByDateAdded(df: pd.DataFrame):
    tmp = df.dropna(subset=["date_added"]).copy()
    tmp["month_added"] = tmp["date_added"].dt.to_period("M").dt.to_timestamp()
    data = tmp.groupby("month_added").size().reset_index(name="count")
    fig = px.area(data, x="month_added", y="count", title="Titles Added to Netflix Over Time")
    fig.update_layout(xaxis_title="Month Added", yaxis_title="Count")
    return fig

def plotByType(df: pd.DataFrame):
    data = df["type"].fillna("Unknown").value_counts().reset_index()
    data.columns = ["Type", "Count"]
    fig = px.bar(data, x="Type", y="Count", title="Movies vs TV Shows", text="Count")
    fig.update_traces(textposition="outside")
    return fig

def plotRatingsDist(df: pd.DataFrame, top_n: int = 12):
    data = df["rating"].fillna("Unknown").value_counts().nlargest(top_n).reset_index()
    data.columns = ["Rating", "Count"]
    fig = px.bar(data, x="Rating", y="Count", title=f"Top {top_n} Ratings Distribution", text="Count")
    fig.update_traces(textposition="outside")
    return fig

def plotDurationDistribution(df: pd.DataFrame):
    tmp = df.copy()
    minutes = tmp["minutes_estimated"].dropna()
    fig = go.Figure()
    if not minutes.empty:
        fig.add_trace(go.Histogram(x=minutes, nbinsx=50, name="Minutes (Movies & TV est.)"))
    tv_seasons = tmp.loc[tmp["duration_type"] == "seasons", "duration_value"].dropna()
    if not tv_seasons.empty:
        fig.add_trace(go.Histogram(x=tv_seasons, nbinsx=20, name="Seasons (TV)", opacity=0.6))
    fig.update_layout(
        barmode="overlay",
        title="Duration Distribution",
        xaxis_title="Minutes (or Seasons if minutes unknown)",
        yaxis_title="Count"
    )
    fig.update_traces(opacity=0.75)
    return fig

def plotTopCountries(df: pd.DataFrame, top_n: int = 15):
    first_country = df["country_list"].apply(lambda lst: lst[0] if isinstance(lst, list) and lst else "Unknown")
    data = first_country.value_counts().nlargest(top_n).reset_index()
    data.columns = ["Country", "Count"]
    fig = px.bar(data, x="Country", y="Count", title=f"Top {top_n} Producing Countries", text="Count")
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plotChoropleth(df: pd.DataFrame):
    tmp = df.explode("country_list")
    tmp["country_list"] = tmp["country_list"].fillna("Unknown")
    counts = tmp.groupby("country_list").size().reset_index(name="Count")
    fig = px.choropleth(
        counts,
        locations="country_list",
        locationmode="country names",
        color="Count",
        title="Global Distribution of Netflix Titles",
        color_continuous_scale="Viridis"
    )
    return fig

def plotGenreHeatmap(exploded_genres: pd.DataFrame, top_n: int = 15):
    top_genres = (
        exploded_genres["genre"].fillna("Unknown")
        .value_counts().nlargest(top_n).index.tolist()
    )
    per_title = (
        exploded_genres[exploded_genres["genre"].isin(top_genres)]
        .groupby("title")["genre"]
        .apply(lambda s: set(s.dropna()))
    )
    genre_index = {g: i for i, g in enumerate(top_genres)}
    mat = [[0]*len(top_genres) for _ in range(len(top_genres))]
    for gset in per_title:
        gset = list(gset)
        for i in range(len(gset)):
            for j in range(i, len(gset)):
                gi, gj = genre_index[gset[i]], genre_index[gset[j]]
                mat[gi][gj] += 1
                if gi != gj:
                    mat[gj][gi] += 1
    heat_df = pd.DataFrame(mat, index=top_genres, columns=top_genres)
    fig = px.imshow(
        heat_df,
        labels=dict(x="Genre", y="Genre", color="Co-occurrence"),
        x=top_genres, y=top_genres,
        title=f"Genre Co-occurrence Heatmap (Top {top_n})"
    )
    return fig

def plotVaderVsSpacy(df: pd.DataFrame, textCol: str = "text"):
    """
    Scatter comparing VADER compound vs spaCy positive prob.
    Colors by agreement of labels.
    """
    if "vader_compound" not in df.columns or "spacy_pos_prob" not in df.columns:
        raise ValueError("Dataframe missing vader_compound or spacy_pos_prob columns.")

    # determine agreement if labels exist
    agree = None
    if "vader_label" in df.columns and "spacy_label" in df.columns:
        agree = (df["vader_label"] == df["spacy_label"]).map({True: "Agree", False: "Disagree"})

    hover = ["title", "type", "release_year"]
    if textCol in df.columns:
        hover.append(textCol)

    fig = px.scatter(
        df,
        x="vader_compound",
        y="spacy_pos_prob",
        color=agree if agree is not None else None,
        hover_data={h: True for h in hover},
        title="VADER vs spaCy sentiment"
    )
    fig.update_layout(
        xaxis_title="VADER compound (neg → pos)",
        yaxis_title="spaCy positive probability"
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
    fig.add_vline(x=0.0, line_dash="dot", line_color="gray")
    return fig

def plotLabelCounts(df: pd.DataFrame, which: str = "spacy_label"):
    """
    Bar chart of label distribution for spaCy or VADER columns.
    """
    if which not in df.columns:
        raise ValueError(f"Column not found: {which}")
    counts = df[which].fillna("Unknown").value_counts().reset_index()
    counts.columns = ["label", "count"]
    fig = px.bar(counts, x="label", y="count", title=f"{which} distribution", text="count")
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="Label", yaxis_title="Count")
    return fig
