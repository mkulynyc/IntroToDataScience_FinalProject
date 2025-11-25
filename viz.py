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


def plotLabelCounts(df: pd.DataFrame, which: str = "spacy_label"):
    """
    Simple bar chart of label counts for the given label column (e.g. 'spacy_label' or 'vader_label').
    """
    if df is None or df.empty:
        return go.Figure()
    if which not in df.columns:
        # fallback: try common names
        if "spacy_label" in df.columns:
            which = "spacy_label"
        elif "vader_label" in df.columns:
            which = "vader_label"
        else:
            return go.Figure()

    counts = df[which].fillna("(missing)").value_counts().reset_index()
    counts.columns = [which, "count"]
    fig = px.bar(counts, x=which, y="count", title=f"Label distribution: {which}", text="count")
    fig.update_traces(textposition="outside")
    return fig


def plotVaderVsSpacy(df: pd.DataFrame, textCol: str = "text"):
    """
    Scatter plot comparing VADER compound score vs spaCy positive probability.
    Expects columns: 'vader_compound' and 'spacy_pos_prob' (or similarly prefixed).
    Hover uses `textCol` if present.
    """
    if df is None or df.empty:
        return go.Figure()

    # find columns
    vader_col = "vader_compound" if "vader_compound" in df.columns else None
    spacy_col = "spacy_pos_prob" if "spacy_pos_prob" in df.columns else None
    if vader_col is None and "vader_compound" in df.columns:
        vader_col = "vader_compound"
    if spacy_col is None and "spacy_pos_prob" in df.columns:
        spacy_col = "spacy_pos_prob"

    if vader_col is None or spacy_col is None:
        # try alternative prefixes
        vader_candidates = [c for c in df.columns if c.endswith("compound")]
        spacy_candidates = [c for c in df.columns if c.endswith("pos_prob") or c.endswith("pos")]
        vader_col = vader_candidates[0] if vader_candidates else None
        spacy_col = spacy_candidates[0] if spacy_candidates else None

    if vader_col is None or spacy_col is None:
        return go.Figure()

    hover = df[textCol].astype(str) if textCol in df.columns else None
    color_col = "spacy_label" if "spacy_label" in df.columns else ("vader_label" if "vader_label" in df.columns else None)

    fig = px.scatter(
        df,
        x=vader_col,
        y=spacy_col,
        color=color_col,
        hover_data={textCol: True} if hover is not None else None,
        labels={vader_col: "VADER compound", spacy_col: "spaCy POS prob"},
        title="VADER (compound) vs spaCy (POS prob)"
    )
    fig.update_traces(marker=dict(opacity=0.7, size=8))
    return fig
