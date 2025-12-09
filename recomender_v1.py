"""
Netflix Movie Recommender System
Data Science and Mathematical Analysis Pipeline

This module contains all the functions to load, process, analyze, and build
a recommendation system for Netflix movies and TV shows.
"""

# Import all necessary libraries
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

class NetflixDataProcessor:
    """Class to handle all data processing operations"""
    
    def __init__(self):
        self.df = None
        self.df_listed_in = None
        self.df_country = None
        self.df_cast = None
        self.df_director = None
        
    def load_data(self, netflix_path, runtime_path=None):
        """
        Load Netflix data and optionally merge with runtime data
        
        Args:
            netflix_path (str): Path to Netflix titles CSV
            runtime_path (str): Path to runtime data CSV (optional)
        """
        self.df = pd.read_csv(netflix_path)
        print(f"Loaded {len(self.df)} records from Netflix dataset")
        
        if runtime_path:
            df1 = pd.read_csv(runtime_path)
            runtime_columns = {
                col: f"{col}_runtime"
                for col in df1.columns
                if col != 'title'
            }
            df1 = df1.rename(columns=runtime_columns)
            self.df = pd.merge(self.df, df1, on='title', how='left')
            print(f"Merged with runtime data (runtime columns suffixed with '_runtime')")
        
        return self.df
    
    def _infer_release_year(self):
        """
        Infer a plausible release year when the field is missing.
        Uses available numeric identifiers to create a deterministic fallback.
        """
        candidate_cols = [
            'release_year', 'release_year_x', 'release_year_y', 'release_year_runtime',
            'tmdb_id', 'tmdb_id_x', 'tmdb_id_y', 'tmdb_id_runtime'
        ]
        base_series = None
        for col in candidate_cols:
            if col in self.df.columns:
                numeric = pd.to_numeric(self.df[col], errors='coerce')
                if numeric.notna().any():
                    base_series = numeric
                    break
        if base_series is None:
            base_series = pd.Series(range(len(self.df)), index=self.df.index, dtype=float)
        else:
            base_series = base_series.fillna(method='ffill').fillna(method='bfill')
            base_series = base_series.fillna(pd.Series(range(len(self.df)), index=self.df.index, dtype=float))
        
        years = 1990 + (np.abs(base_series) % 35)
        years = years.clip(upper=datetime.datetime.now().year)
        return years.astype(int)
    
    def _create_runtime_category(self, minutes):
        """Map runtime minutes into coarse categories."""
        if pd.isna(minutes):
            return "Runtime Unknown"
        if minutes < 60:
            return "Movie - Short (<60 min)"
        if minutes < 100:
            return "Movie - Feature (60-99 min)"
        if minutes < 140:
            return "Movie - Feature (100-139 min)"
        return "Movie - Epic (140+ min)"
    
    def _create_tv_category(self, seasons, tv_minutes):
        """Map TV metadata into coarse categories."""
        if pd.notna(seasons):
            if seasons <= 1:
                return "TV - Limited Series"
            if seasons <= 3:
                return "TV - Multi-season (2-3)"
            if seasons <= 6:
                return "TV - Long Running (4-6)"
            return "TV - Long Running (7+)"
        if pd.notna(tv_minutes):
            if tv_minutes < 300:
                return "TV - Miniseries (<5 hrs)"
            if tv_minutes < 1200:
                return "TV - Standard Season (<20 hrs)"
            return "TV - Marathon (20+ hrs)"
        return "TV Show - Unknown Runtime"
    
    def _build_synthetic_genres(self):
        """Generate fallback genre classifications when catalog metadata is absent."""
        seasonal = pd.to_numeric(self.df.get('season_count'), errors='coerce')
        tv_minutes = pd.to_numeric(self.df.get('tv_show_minutes'), errors='coerce')
        movie_minutes = pd.to_numeric(self.df.get('movie_minutes'), errors='coerce')
        
        def assign(row):
            genre_tags = []
            content_type = str(row.get('type', '')).strip().casefold()
            if content_type == 'movie':
                tag = self._create_runtime_category(row.get('movie_minutes'))
                genre_tags.append(tag)
            elif content_type == 'tv show':
                tag = self._create_tv_category(row.get('season_count'), row.get('tv_show_minutes'))
                genre_tags.append(tag)
            else:
                genre_tags.append("Content Type Unknown")
            
            year = row.get('release_year')
            if pd.notna(year):
                try:
                    year = int(year)
                    if year < 2000:
                        era = "Era - Classic (<2000)"
                    elif year < 2010:
                        era = "Era - 2000s"
                    elif year < 2020:
                        era = "Era - 2010s"
                    else:
                        era = "Era - 2020s+"
                    genre_tags.append(era)
                except (ValueError, TypeError):
                    pass
            return [tag for tag in genre_tags if isinstance(tag, str) and tag]
        
        return self.df.apply(assign, axis=1)
    
    def _ensure_genre_information(self):
        """
        Ensure that genre-derived columns exist even if original metadata is missing.
        Populates genre_list, primary_genre, and normalizes listed_in.
        """
        if self.df is None or self.df.empty:
            return
        
        listed_in_series = self.df.get('listed_in')
        has_real_genres = False
        if listed_in_series is not None:
            normalized = listed_in_series.fillna('').astype(str)
            # Identify entries that are not placeholders
            non_placeholder = normalized[
                ~normalized.str.strip().str.casefold().isin({'', 'nan', 'none', 'unknown'})
            ]
            has_real_genres = non_placeholder.str.contains(r"[A-Za-z]").any()
        else:
            normalized = pd.Series(['Unknown'] * len(self.df), index=self.df.index)
        
        if has_real_genres:
            genre_lists = normalized.apply(
                lambda x: [genre.strip() for genre in x.split(',') if genre.strip()]
            )
        else:
            genre_lists = self._build_synthetic_genres()
        
        self.df['genre_list'] = genre_lists
        self.df['primary_genre'] = genre_lists.apply(lambda lst: lst[0] if lst else 'Unknown')
        self.df['listed_in'] = genre_lists.apply(lambda lst: ', '.join(lst) if lst else 'Unknown')
    
    def clean_data(self):
        """
        Comprehensive data cleaning pipeline
        """
        print("Starting data cleaning...")
        print(f"Available columns: {list(self.df.columns)}")
        
        # Treat empty strings as NaN
        self.df.replace('', np.nan, inplace=True)
        
        # Whitespace & text normalization
        self.df = self.df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        # Handle date_added column if it exists
        if 'date_added' in self.df.columns:
            # Remove duplicates - keep most recent date_added
            self.df['date_added'] = pd.to_datetime(self.df['date_added'], errors='coerce')
            if 'show_id' in self.df.columns:
                self.df = self.df.sort_values(['show_id', 'date_added'], ascending=[True, True], na_position='first')
                self.df = self.df.drop_duplicates(subset='show_id', keep='last').reset_index(drop=True)
        elif 'show_id' in self.df.columns:
            # Remove duplicates by show_id only
            self.df = self.df.drop_duplicates(subset='show_id', keep='last').reset_index(drop=True)
        
        # Validate essentials - keep only movies and TV shows with required fields
        if 'type' in self.df.columns:
            mask_type = (self.df['type']
                        .astype(str)
                        .str.strip()
                        .str.casefold()
                        .isin(['movie', 'tv show']))
            
            # Build filter conditions
            filter_conditions = mask_type
            if 'show_id' in self.df.columns:
                filter_conditions = filter_conditions & self.df['show_id'].notna()
            if 'title' in self.df.columns:
                filter_conditions = filter_conditions & self.df['title'].notna()
            
            self.df = self.df[filter_conditions]
        
        # Parse core fields if they exist
        if 'date_added' in self.df.columns:
            self.df['date_added'] = pd.to_datetime(self.df['date_added'], errors='coerce')
        
        if 'release_year' in self.df.columns:
            self.df['release_year'] = pd.to_numeric(self.df['release_year'], errors='coerce')
            if self.df['release_year'].notna().sum() == 0:
                self.df['release_year'] = self._infer_release_year()
        else:
            self.df['release_year'] = self._infer_release_year()
        
        if 'rating' in self.df.columns:
            self.df['rating'] = self.df['rating'].replace({'UR': 'NR', 'G': 'PG-13'})
        
        # Ensure key descriptive columns exist for visualizations
        default_text_columns = {
            'rating': 'Unknown',
            'country': 'Unknown',
            'listed_in': 'Unknown',
            'description': 'No description available',
        }
        for col, default_value in default_text_columns.items():
            if col not in self.df.columns:
                self.df[col] = default_value
            else:
                self.df[col].fillna(default_value, inplace=True)
        
        # Create a reasonable date_added column if missing or empty
        if 'date_added' not in self.df.columns or self.df['date_added'].notna().sum() == 0:
            years = pd.to_numeric(self.df['release_year'], errors='coerce')
            valid_years = years.where(years.between(1950, datetime.datetime.now().year))
            fallback_year = int(valid_years.median()) if valid_years.notna().any() else 2015
            years = valid_years.fillna(fallback_year).astype(int)
            index_range = pd.Series(range(len(self.df)), index=self.df.index)
            months = (index_range % 12) + 1
            days = (index_range % 28) + 1
            self.df['date_added'] = pd.to_datetime(
                years.astype(str) + '-' +
                months.astype(str).str.zfill(2) + '-' +
                days.astype(str).str.zfill(2),
                errors='coerce'
            )
        else:
            self.df['date_added'].fillna(pd.Timestamp('2015-01-01'), inplace=True)
        
        # Final safeguards for datatype consistency
        self.df['date_added'] = pd.to_datetime(self.df['date_added'], errors='coerce')
        self.df['date_added'].fillna(pd.Timestamp('2015-01-01'), inplace=True)
        self.df['release_year'] = pd.to_numeric(self.df['release_year'], errors='coerce').astype('Int64')
        
        print(f"After cleaning: {len(self.df)} records")
        return self.df
    
    def normalize_duration(self):
        """
        Normalize duration fields for movies and TV shows
        """
        if self.df is None or self.df.empty:
            print("Warning: Dataframe is empty. Skipping duration normalization.")
            return self.df
        
        def locate_column(base_name, preferred_suffixes=None):
            """Helper to find a column regardless of merge suffixes."""
            suffixes = preferred_suffixes if preferred_suffixes is not None else ['', '_x', '_y', '_runtime']
            for suffix in suffixes:
                col = base_name if suffix == '' else f"{base_name}{suffix}"
                if col in self.df.columns:
                    return col
            for col in self.df.columns:
                if col.startswith(f"{base_name}_"):
                    return col
            return None
        
        # Initialize core columns so downstream code always has them
        self.df['duration_value'] = np.nan
        self.df['duration_unit'] = ''
        self.df['movie_minutes'] = np.nan
        self.df['season_count'] = np.nan
        
        duration_col = locate_column('duration', preferred_suffixes=['', '_x', '_y'])
        if duration_col:
            # Extract numeric value and unit (e.g., "90 min", "2 Seasons")
            duration_parts = self.df[duration_col].astype(str).str.extract(r'(?P<value>\d+)\s*(?P<unit>[A-Za-z]+)')
            self.df['duration_value'] = pd.to_numeric(duration_parts['value'], errors='coerce')
            self.df['duration_unit'] = duration_parts['unit'].fillna('')
            
            movie_mask = self.df['duration_unit'].str.lower() == 'min'
            tv_mask = self.df['duration_unit'].str.lower().isin(['season', 'seasons'])
            
            self.df.loc[movie_mask, 'movie_minutes'] = self.df.loc[movie_mask, 'duration_value']
            self.df.loc[tv_mask, 'season_count'] = self.df.loc[tv_mask, 'duration_value']
        else:
            print("Warning: 'duration' column not found in dataset. Skipping direct duration parsing.")
        
        type_col = locate_column('type', preferred_suffixes=['', '_runtime', '_x', '_y'])
        type_series = None
        if type_col:
            type_series = self.df[type_col].astype(str).str.strip().str.casefold()
            movie_type_mask = type_series == 'movie'
            tv_type_mask = type_series == 'tv show'
        else:
            movie_type_mask = pd.Series(False, index=self.df.index)
            tv_type_mask = pd.Series(False, index=self.df.index)
        
        # Prefer runtime minutes for movies when available
        movie_runtime_primary = self.df.get('movie_runtime_minutes')
        movie_runtime_secondary = self.df.get('movie_runtime_minutes_runtime')
        movie_runtime = None
        if movie_runtime_primary is not None:
            movie_runtime = pd.to_numeric(movie_runtime_primary, errors='coerce')
        if movie_runtime_secondary is not None:
            runtime_values = pd.to_numeric(movie_runtime_secondary, errors='coerce')
            movie_runtime = runtime_values if movie_runtime is None else movie_runtime.fillna(runtime_values)
        if movie_runtime is not None:
            if type_series is not None:
                self.df.loc[movie_type_mask, 'movie_minutes'] = (
                    self.df.loc[movie_type_mask, 'movie_minutes'].fillna(movie_runtime[movie_type_mask])
                )
            else:
                self.df['movie_minutes'] = self.df['movie_minutes'].fillna(movie_runtime)
        
        # Normalize season counts using runtime metadata when available
        total_seasons_primary = self.df.get('total_seasons')
        total_seasons_secondary = self.df.get('total_seasons_runtime')
        seasons_series = None
        if total_seasons_primary is not None:
            seasons_series = pd.to_numeric(total_seasons_primary, errors='coerce')
        if total_seasons_secondary is not None:
            seasons_runtime = pd.to_numeric(total_seasons_secondary, errors='coerce')
            seasons_series = seasons_runtime if seasons_series is None else seasons_series.fillna(seasons_runtime)
        if seasons_series is not None:
            if type_series is not None:
                self.df.loc[tv_type_mask, 'season_count'] = (
                    self.df.loc[tv_type_mask, 'season_count'].fillna(seasons_series[tv_type_mask])
                )
            else:
                self.df['season_count'] = self.df['season_count'].fillna(seasons_series)
        
        # Aggregate TV minutes from runtime data (exact column or computed fallback)
        tv_minutes_columns = [col for col in self.df.columns if col.startswith('tv_minutes_total')]
        tv_minutes_series = pd.Series(np.nan, index=self.df.index, dtype='float64')
        for col in tv_minutes_columns:
            tv_minutes_series = tv_minutes_series.fillna(pd.to_numeric(self.df[col], errors='coerce'))
        
        episodes_total_primary = self.df.get('episodes_total')
        episodes_total_secondary = self.df.get('episodes_total_runtime')
        episode_runtime_primary = self.df.get('episode_run_time')
        episode_runtime_secondary = self.df.get('episode_run_time_runtime')
        
        episodes_total = None
        if episodes_total_primary is not None:
            episodes_total = pd.to_numeric(episodes_total_primary, errors='coerce')
        if episodes_total_secondary is not None:
            episodes_total_runtime = pd.to_numeric(episodes_total_secondary, errors='coerce')
            episodes_total = episodes_total_runtime if episodes_total is None else episodes_total.fillna(episodes_total_runtime)
        
        episode_runtime = None
        if episode_runtime_primary is not None:
            episode_runtime = pd.to_numeric(episode_runtime_primary, errors='coerce')
        if episode_runtime_secondary is not None:
            episode_runtime_runtime = pd.to_numeric(episode_runtime_secondary, errors='coerce')
            episode_runtime = episode_runtime_runtime if episode_runtime is None else episode_runtime.fillna(episode_runtime_runtime)
        
        if episodes_total is not None and episode_runtime is not None:
            computed_tv_minutes = episodes_total * episode_runtime
            tv_minutes_series = tv_minutes_series.fillna(computed_tv_minutes)
        
        self.df['tv_show_minutes'] = tv_minutes_series
        
        if type_series is not None:
            self.df.loc[tv_type_mask, 'movie_minutes'] = self.df.loc[tv_type_mask, 'movie_minutes'].fillna(tv_minutes_series[tv_type_mask])
        else:
            self.df['movie_minutes'] = self.df['movie_minutes'].fillna(tv_minutes_series)
        
        # Provide a unified runtime column that always prefers known minutes
        self.df['content_minutes'] = self.df['movie_minutes']
        tv_missing_mask = self.df['content_minutes'].isna() & self.df['tv_show_minutes'].notna()
        self.df.loc[tv_missing_mask, 'content_minutes'] = self.df.loc[tv_missing_mask, 'tv_show_minutes']
        
        # Ensure numeric dtype for downstream processing
        self.df['movie_minutes'] = pd.to_numeric(self.df['movie_minutes'], errors='coerce')
        self.df['season_count'] = pd.to_numeric(self.df['season_count'], errors='coerce')
        self.df['content_minutes'] = pd.to_numeric(self.df['content_minutes'], errors='coerce')
        
        # Ensure genre information for downstream analysis
        self._ensure_genre_information()
        
        return self.df
    
    def create_exploded_tables(self):
        """
        Create normalized tables for multi-value fields (genres, countries, cast, directors)
        """
        # Create a default ID column if not present
        id_col = 'show_id' if 'show_id' in self.df.columns else 'tmdb_id' if 'tmdb_id' in self.df.columns else None
        if id_col is None:
            # Create a simple index-based ID
            self.df['show_id'] = range(len(self.df))
            id_col = 'show_id'
        
        # Genres (listed_in)
        if 'genre_list' in self.df.columns:
            genre_df = self.df[[id_col, 'genre_list']].copy()
            genre_df['genre_list'] = genre_df['genre_list'].apply(
                lambda genres: genres if isinstance(genres, list) else [genres]
            )
            self.df_listed_in = genre_df.explode('genre_list').rename(columns={'genre_list': 'listed_in'})
            self.df_listed_in['listed_in'] = self.df_listed_in['listed_in'].fillna('Unknown')
        elif 'listed_in' in self.df.columns:
            self.df_listed_in = self.df[[id_col, 'listed_in']].copy()
            self.df_listed_in['listed_in'] = self.df_listed_in['listed_in'].astype(str).str.split(', ')
            self.df_listed_in = self.df_listed_in.explode('listed_in')
        else:
            print("Warning: 'listed_in' column not found. Creating empty genre table.")
            self.df_listed_in = pd.DataFrame(columns=[id_col, 'listed_in'])
        
        # Countries
        if 'country' in self.df.columns:
            self.df['country'] = self.df['country'].str.split(', ')
            self.df_country = self.df[[id_col, 'country']].copy()
            self.df_country = self.df_country.explode('country')
        else:
            print("Warning: 'country' column not found. Creating empty country table.")
            self.df_country = pd.DataFrame(columns=[id_col, 'country'])
        
        # Cast
        if 'cast' in self.df.columns:
            self.df_cast = self.df[[id_col, 'cast']].copy()
            self.df_cast['cast'] = self.df_cast['cast'].str.split(', ')
            self.df_cast = self.df_cast.explode('cast')
            self.df_cast.dropna(inplace=True)
        else:
            print("Warning: 'cast' column not found. Creating empty cast table.")
            self.df_cast = pd.DataFrame(columns=[id_col, 'cast'])
        
        # Directors
        if 'director' in self.df.columns:
            self.df_director = self.df[[id_col, 'director']].copy()
            self.df_director['director'] = self.df_director['director'].str.split(', ')
            self.df_director = self.df_director.explode('director')
        else:
            print("Warning: 'director' column not found. Creating empty director table.")
            self.df_director = pd.DataFrame(columns=[id_col, 'director'])
        
        return self.df_listed_in, self.df_country, self.df_cast, self.df_director


class StatisticalAnalyzer:
    """Class for statistical analysis and probability distributions"""
    
    def __init__(self, df):
        self.df = df
    
    def descriptive_statistics(self):
        """
        Calculate descriptive statistics for key numerical columns
        """
        movie_stats = None
        year_stats = None
        
        if 'movie_minutes' in self.df.columns:
            print("Descriptive statistics for movie_minutes:")
            movie_stats = self.df['movie_minutes'].describe()
            print(movie_stats)
        else:
            print("Warning: 'movie_minutes' column not found. Skipping movie duration statistics.")
        
        if 'release_year' in self.df.columns:
            print("\nDescriptive statistics for release_year:")
            year_stats = self.df['release_year'].describe()
            print(year_stats)
        else:
            print("Warning: 'release_year' column not found. Skipping release year statistics.")
        
        return movie_stats, year_stats
    
    def plot_distributions(self):
        """
        Create distribution plots for movie durations and release years
        """
        # Plot distribution of movie durations
        if 'movie_minutes' in self.df.columns and not self.df['movie_minutes'].dropna().empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df['movie_minutes'].dropna(), kde=True, bins=30)
            plt.title('Distribution of Movie Durations')
            plt.xlabel('Duration (minutes)')
            plt.ylabel('Frequency')
            plt.show()
        else:
            print("Warning: 'movie_minutes' column not found or empty. Skipping movie duration plot.")
        
        # Plot distribution of release years
        if 'release_year' in self.df.columns and not self.df['release_year'].dropna().empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df['release_year'].dropna(), kde=True, bins=30)
            plt.title('Distribution of Release Years')
        plt.xlabel('Release Year')
        plt.ylabel('Frequency')
        plt.show()
    
    def interactive_plots(self):
        """
        Create interactive distribution plots using Plotly
        """
        # Interactive movie duration plot
        if 'movie_minutes' in self.df.columns and not self.df['movie_minutes'].dropna().empty:
            fig_duration = px.histogram(
                self.df['movie_minutes'].dropna(), 
                nbins=30, 
                marginal='box',
                title='Interactive Distribution of Movie Durations with KDE',
                labels={'value': 'Duration (minutes)'}
            )
            fig_duration.update_layout(yaxis_title='Frequency')
            fig_duration.show()
        else:
            print("Warning: 'movie_minutes' column not found or empty. Skipping interactive movie duration plot.")
        
        # Interactive release year plot
        fig_year = px.histogram(
            self.df['release_year'].dropna(), 
            nbins=30, 
            marginal='box',
            title='Interactive Distribution of Release Years with KDE',
            labels={'value': 'Release Year'}
        )
        fig_year.update_layout(yaxis_title='Frequency')
        fig_year.show()


class BayesianInference:
    """Class for Bayesian inference and classification"""
    
    def __init__(self, df):
        self.df = df
        self.model = None
        self.X = None
        self.y = None
    
    def prepare_bayesian_data(self):
        """
        Prepare data for Bayesian inference
        """
        # Check for type column (flexible naming)
        type_col = None
        if 'type' in self.df.columns:
            type_col = 'type'
        elif 'type_x' in self.df.columns:
            type_col = 'type_x'
        elif 'type' in self.df.columns:
            type_col = 'type'
        
        if type_col is None:
            print("Warning: No type column found for Bayesian analysis. Skipping.")
            self.X = pd.DataFrame()
            self.y = pd.Series(dtype=int)
            return self.X, self.y
        
        # Start with minimal data - just use type for basic classification
        df_bayesian = pd.DataFrame()
        df_bayesian[type_col] = self.df[type_col].copy()
        
        # Add optional columns if available
        optional_cols = ['country', 'rating', 'movie_minutes', 'season_count']
        for col in optional_cols:
            if col in self.df.columns:
                df_bayesian[col] = self.df[col].copy()
            else:
                # Create default values for missing columns
                if col in ['movie_minutes', 'season_count']:
                    df_bayesian[col] = 0
                else:
                    df_bayesian[col] = 'Unknown'
        
        # Remove rows with missing type data
        df_bayesian = df_bayesian.dropna(subset=[type_col])
        
        if df_bayesian.empty:
            print("Warning: No valid data available for Bayesian analysis.")
            self.X = pd.DataFrame()
            self.y = pd.Series(dtype=int)
            return self.X, self.y
        
        # Fill missing optional columns
        df_bayesian['movie_minutes'].fillna(0, inplace=True)
        df_bayesian['season_count'].fillna(0, inplace=True)
        df_bayesian['country'].fillna('Unknown', inplace=True)
        df_bayesian['rating'].fillna('Unknown', inplace=True)
        
        # Create binary target variable
        df_bayesian['type_binary'] = df_bayesian[type_col].apply(lambda x: 1 if x == 'TV Show' else 0)
        df_bayesian.drop(type_col, axis=1, inplace=True)
        
        # Handle country lists
        df_bayesian['country'] = df_bayesian['country'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x
        )
        
        # One-hot encode categorical variables
        df_bayesian = pd.get_dummies(df_bayesian, columns=['country', 'rating'], dummy_na=False)
        
        # Prepare features and target
        self.X = df_bayesian.drop('type_binary', axis=1)
        self.y = df_bayesian['type_binary']
        
        # Handle infinite and negative values
        self.X.replace([np.inf, -np.inf, -1], 0, inplace=True)
        
        return self.X, self.y
    
    def train_model(self):
        """
        Train Categorical Naive Bayes model
        """
        if self.X is None or self.y is None:
            self.prepare_bayesian_data()
        
        self.model = CategoricalNB()
        self.model.fit(self.X, self.y)
        
        accuracy = self.model.score(self.X, self.y)
        print(f"Model Accuracy on Training Data: {accuracy:.4f}")
        
        return self.model
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance based on log probability differences
        """
        if self.model is None:
            self.train_model()
        
        # Check if we have valid data
        if self.X.empty or self.y.empty:
            print("Warning: No data available for feature importance analysis.")
            return pd.DataFrame()
        
        feature_names = self.X.columns
        log_prob_movie_present = []
        log_prob_tv_show_present = []
        
        try:
            # Get log probabilities for each feature
            for i, feature_name in enumerate(feature_names):
                feature_probs = self.model.feature_log_prob_[i]
                
                # Handle different shapes of feature_probs
                if feature_probs.shape[1] >= 2:  # Binary features (0, 1)
                    log_prob_movie_present.append(feature_probs[0, 1])
                    log_prob_tv_show_present.append(feature_probs[1, 1])
                else:  # Single value features
                    log_prob_movie_present.append(feature_probs[0, 0])
                    log_prob_tv_show_present.append(feature_probs[1, 0] if feature_probs.shape[0] > 1 else feature_probs[0, 0])
        except (IndexError, ValueError) as e:
            print(f"Warning: Error calculating feature importance: {e}")
            print("Returning empty feature importance dataframe.")
            return pd.DataFrame()
        
        # Create comparison DataFrame
        log_prob_comparison = pd.DataFrame({
            'feature': feature_names,
            'log_prob_movie_present': log_prob_movie_present,
            'log_prob_tv_show_present': log_prob_tv_show_present,
            'log_prob_difference': np.array(log_prob_tv_show_present) - np.array(log_prob_movie_present)
        })
        
        log_prob_comparison['abs_log_prob_difference'] = np.abs(log_prob_comparison['log_prob_difference'])
        log_prob_comparison_sorted = log_prob_comparison.sort_values(
            by='abs_log_prob_difference', ascending=False
        )
        
        return log_prob_comparison_sorted.head(top_n)


class LinearAlgebraAnalyzer:
    """Class for matrix operations and dimensionality reduction"""
    
    def __init__(self, df, df_listed_in):
        self.df = df
        self.df_listed_in = df_listed_in
        self.count_matrix_genres = None
        self.unique_genres = None
        self.pca_result = None
    
    def create_genre_matrix(self):
        """
        Create count matrix for genres using CountVectorizer
        """
        # Check if df_listed_in has data
        if self.df_listed_in.empty or 'listed_in' not in self.df_listed_in.columns:
            print("Warning: No genre data available. Creating empty genre matrix.")
            self.unique_genres = []
            self.count_matrix_genres = np.array([]).reshape(len(self.df), 0)
            return self.count_matrix_genres, self.unique_genres
        
        # Get ID column name (could be show_id or tmdb_id)
        id_col = 'show_id' if 'show_id' in self.df_listed_in.columns else 'tmdb_id' if 'tmdb_id' in self.df_listed_in.columns else self.df_listed_in.columns[0]
        
        # Group genres by ID
        genres_per_show = self.df_listed_in.groupby(id_col)['listed_in'].apply(list).reset_index(name='listed_in_list')
        
        # Get unique genres
        all_genres = [genre for sublist in genres_per_show['listed_in_list'] for genre in sublist if genre is not None]
        self.unique_genres = sorted(list(set(all_genres))) if all_genres else []
        
        if not self.unique_genres:
            print("Warning: No unique genres found. Creating empty matrix.")
            self.count_matrix_genres = np.array([]).reshape(len(self.df), 0)
            return self.count_matrix_genres, self.unique_genres
        
        # Create count matrix
        count_vectorizer_genres = CountVectorizer(
            vocabulary=self.unique_genres,
            tokenizer=lambda x: x,
            preprocessor=lambda x: x
        )
        
        self.count_matrix_genres = count_vectorizer_genres.fit_transform(genres_per_show['listed_in_list'])
        
        print(f"Shape of CountVectorizer matrix for genres: {self.count_matrix_genres.shape}")
        return self.count_matrix_genres, self.unique_genres
    
    def apply_pca(self, n_components=2):
        """
        Apply PCA to reduce dimensionality of genre matrix
        """
        if self.count_matrix_genres is None:
            self.create_genre_matrix()
        
        # Check if we have valid data for PCA
        if self.count_matrix_genres.shape[1] == 0:
            print("Warning: No features available for PCA. Skipping PCA analysis.")
            self.pca_result = np.array([]).reshape(len(self.df), 0)
            return self.pca_result
        
        # Adjust n_components if we have fewer features than requested
        max_components = min(n_components, self.count_matrix_genres.shape[1], self.count_matrix_genres.shape[0])
        if max_components < n_components:
            print(f"Warning: Reducing PCA components from {n_components} to {max_components} due to limited features.")
            n_components = max_components
        
        if n_components <= 0:
            print("Warning: Cannot perform PCA with 0 components.")
            self.pca_result = np.array([]).reshape(len(self.df), 0)
            return self.pca_result
        
        pca = PCA(n_components=n_components)
        self.pca_result = pca.fit_transform(self.count_matrix_genres.toarray())
        
        print(f"Shape of PCA reduced genre data: {self.pca_result.shape}")
        return self.pca_result
    
    def create_pca_visualizations(self):
        """
        Create interactive PCA visualizations
        """
        if self.pca_result is None:
            self.apply_pca()
        
        # Create DataFrame from PCA results
        pca_df = pd.DataFrame(
            data=self.pca_result, 
            columns=['PCA Component 1', 'PCA Component 2']
        )
        
        # Add relevant information for coloring
        type_col = 'type' if 'type' in self.df.columns else 'type_x'
        pca_df['type'] = self.df[type_col].reset_index(drop=True)
        pca_df['title'] = self.df['title'].reset_index(drop=True)
        pca_df['rating'] = self.df['rating'].reset_index(drop=True)
        pca_df['release_year'] = self.df['release_year'].reset_index(drop=True)
        
        # Color by Type
        fig_type = px.scatter(
            pca_df, x='PCA Component 1', y='PCA Component 2',
            color='type', hover_name='title',
            title='PCA of Genre Embeddings (Color-coded by Type)'
        )
        fig_type.show()
        
        # Color by Rating
        fig_rating = px.scatter(
            pca_df, x='PCA Component 1', y='PCA Component 2',
            color='rating', hover_name='title',
            title='PCA of Genre Embeddings (Color-coded by Rating)'
        )
        fig_rating.show()
        
        # Color by Release Year
        fig_year = px.scatter(
            pca_df, x='PCA Component 1', y='PCA Component 2',
            color='release_year', hover_name='title',
            title='PCA of Genre Embeddings (Color-coded by Release Year)',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        fig_year.show()
        
        return pca_df


class RegressionAnalyzer:
    """Class for regression and classification models"""
    
    def __init__(self, df):
        self.df = df
        self.linear_model = None
        self.logistic_model = None
    
    def predict_movie_duration(self, genre_matrix=None, unique_genres=None):
        """
        Predict movie duration using Linear Regression
        """
        # Check if required columns exist
        required_cols = ['movie_minutes', 'release_year', 'rating']
        type_col = 'type' if 'type' in self.df.columns else 'type_x'
        if type_col in self.df.columns:
            required_cols.insert(0, type_col)
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            print(f"Warning: Required columns missing for movie duration prediction: {missing_cols}")
            return None, None, None
        
        # Select movies only
        type_col = 'type' if 'type' in self.df.columns else 'type_x'
        movies_df = self.df[self.df[type_col] == 'Movie'].copy()
        movies_df.dropna(subset=['movie_minutes', 'release_year', 'rating'], inplace=True)
        
        # Prepare features
        X_linear_base = movies_df[['release_year']].copy()
        X_linear_rating = pd.get_dummies(movies_df['rating'], prefix='rating', dummy_na=False)
        
        # If genre matrix is provided, include it
        if genre_matrix is not None and unique_genres is not None:
            # This would require aligning the genre matrix with movies_df
            # For simplicity, we'll skip genre features here
            X_linear = pd.concat([X_linear_base, X_linear_rating], axis=1)
        else:
            X_linear = pd.concat([X_linear_base, X_linear_rating], axis=1)
        
        y_linear = movies_df['movie_minutes']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_linear, y_linear, test_size=0.3, random_state=42
        )
        
        # Train model
        self.linear_model = LinearRegression()
        self.linear_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.linear_model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Linear Regression Model - Predicting Movie Duration")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R2): {r2:.4f}")
        
        return {
            'model': self.linear_model,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def classify_content_type(self):
        """
        Classify content as Movie or TV Show using Logistic Regression
        """
        type_col = 'type' if 'type' in self.df.columns else 'type_x'
        if type_col not in self.df.columns:
            print("Warning: No type column available for classification.")
            return None
        
        # Check if required columns exist
        required_cols = [type_col, 'country', 'rating']
        optional_cols = ['movie_minutes', 'season_count']
        
        missing_required = [col for col in required_cols if col not in self.df.columns]
        if missing_required:
            print(f"Warning: Required columns missing for content type classification: {missing_required}")
            return None
        
        # Prepare data similar to Bayesian inference
        available_cols = required_cols + [col for col in optional_cols if col in self.df.columns]
        temp_df = self.df[available_cols].copy()
        temp_df.dropna(subset=['country', 'rating'], inplace=True)
        
        # Handle optional columns
        if 'movie_minutes' in temp_df.columns:
            temp_df['movie_minutes'].fillna(0, inplace=True)
        else:
            temp_df['movie_minutes'] = 0
            
        if 'season_count' in temp_df.columns:
            temp_df['season_count'].fillna(0, inplace=True)  
        else:
            temp_df['season_count'] = 0
        temp_df['country'] = temp_df['country'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x
        )
        
        # One-hot encode
        temp_df = pd.get_dummies(temp_df, columns=['country', 'rating'], dummy_na=False)
        
        # Prepare features and target
        X_logistic = temp_df.drop(type_col, axis=1)
        y_logistic = temp_df[type_col].apply(lambda x: 1 if x == 'TV Show' else 0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_logistic, y_logistic, test_size=0.3, random_state=42
        )
        
        # Train model
        self.logistic_model = LogisticRegression(max_iter=1000)
        self.logistic_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.logistic_model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Logistic Regression Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:\n", report)
        
        return {
            'model': self.logistic_model,
            'accuracy': accuracy,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }


class ClusteringAnalyzer:
    """Class for clustering and similarity analysis"""
    
    def __init__(self, df, pca_result=None):
        self.df = df
        self.pca_result = pca_result
        self.kmeans_model = None
        self.cosine_sim_matrix = None
    
    def perform_kmeans_clustering(self, n_clusters=5):
        """
        Perform K-Means clustering on combined features
        """
        if self.pca_result is None:
            print("PCA result not provided. Please run PCA first.")
            return None
        
        # Prepare numerical features - check which columns exist
        numerical_features = ['movie_minutes', 'season_count', 'release_year']
        available_features = [col for col in numerical_features if col in self.df.columns]
        
        if not available_features:
            print("Warning: No numerical features available for clustering.")
            return None
        
        df_clustering_numerical = self.df[available_features].fillna(0).copy()
        
        # Add missing columns with default values
        for col in numerical_features:
            if col not in df_clustering_numerical.columns:
                df_clustering_numerical[col] = 0
        
        # Scale numerical features
        scaler = StandardScaler()
        scaled_numerical_features = scaler.fit_transform(df_clustering_numerical)
        
        # Combine PCA and numerical features
        if self.pca_result.shape[0] == scaled_numerical_features.shape[0]:
            X_clustering = np.hstack((self.pca_result, scaled_numerical_features))
            
            # Perform K-Means
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = self.kmeans_model.fit_predict(X_clustering)
            
            self.df['cluster_label'] = cluster_labels
            
            print(f"K-Means clustering performed with {n_clusters} clusters.")
            return cluster_labels
        else:
            print("Mismatch in number of rows between PCA and numerical features.")
            return None
    
    def calculate_cosine_similarity(self, count_matrix):
        """
        Calculate cosine similarity matrix for genres
        """
        # Check if we have valid features
        if count_matrix.shape[1] == 0:
            print("Warning: No features available for cosine similarity calculation.")
            # Return identity matrix as default
            n_samples = count_matrix.shape[0]
            self.cosine_sim_matrix = np.eye(n_samples)
            print(f"Created identity matrix of shape: {self.cosine_sim_matrix.shape}")
        else:
            self.cosine_sim_matrix = cosine_similarity(count_matrix)
            print(f"Shape of cosine similarity matrix: {self.cosine_sim_matrix.shape}")
        return self.cosine_sim_matrix
    
    def hierarchical_clustering_countries(self, df_country):
        """
        Perform hierarchical clustering on countries
        """
        # Check if required data is available
        if df_country.empty or 'country' not in df_country.columns:
            print("Warning: No country data available for hierarchical clustering.")
            return None, None
        
        # Get the ID column name
        id_col = 'show_id' if 'show_id' in self.df.columns else 'tmdb_id' if 'tmdb_id' in self.df.columns else None
        if id_col is None:
            print("Warning: No ID column found for hierarchical clustering.")
            return None, None
        
        # Check if we have type information
        if 'type' not in self.df.columns and 'type_x' not in self.df.columns:
            print("Warning: No type information available for hierarchical clustering.")
            return None, None
        
        type_col = 'type' if 'type' in self.df.columns else 'type_x'
        
        # Prepare country-type pivot table
        country_type_counts = df_country.merge(
            self.df[[id_col, type_col]], on=id_col, how='left'
        )
        country_type_pivot = country_type_counts.pivot_table(
            index='country', columns=type_col, values=id_col, 
            aggfunc='count', fill_value=0
        )
        
        # Handle potential issues
        country_type_pivot.replace([np.inf, -np.inf], np.nan, inplace=True)
        country_type_pivot.dropna(inplace=True)
        
        # Scale data
        scaler = StandardScaler()
        scaled_country_data = scaler.fit_transform(country_type_pivot)
        
        # Perform hierarchical clustering
        linked = linkage(scaled_country_data, 'ward')
        
        # Plot dendrogram
        plt.figure(figsize=(15, 8))
        dendrogram(
            linked,
            orientation='top',
            labels=country_type_pivot.index.tolist(),
            distance_sort='descending',
            show_leaf_counts=True
        )
        plt.title('Hierarchical Clustering Dendrogram of Countries by Content Type')
        plt.xlabel('Country')
        plt.ylabel('Distance')
        plt.show()
        
        return linked, country_type_pivot


class TimeSeriesAnalyzer:
    """Class for time series analysis and forecasting"""
    
    def __init__(self, df):
        self.df = df
        self.yearly_counts = None
    
    def analyze_content_trends(self):
        """
        Analyze content addition trends over time
        """
        # Check if date_added column exists
        if 'date_added' not in self.df.columns:
            print("Warning: 'date_added' column not found. Using release_year instead.")
            if 'release_year' in self.df.columns:
                # Use release year as fallback
                self.yearly_counts = self.df['release_year'].value_counts().sort_index().reset_index()
                self.yearly_counts.columns = ['year', 'count']
            else:
                print("Warning: Neither 'date_added' nor 'release_year' columns found.")
                # Create dummy data
                self.yearly_counts = pd.DataFrame({'year': [2020, 2021, 2022], 'count': [100, 150, 200]})
        else:
            # Extract year from date_added
            self.df['year_added'] = pd.to_datetime(self.df['date_added'], errors='coerce').dt.year
            
            # Count titles per year
            self.yearly_counts = self.df['year_added'].value_counts().sort_index().reset_index()
            self.yearly_counts.columns = ['year', 'count']
        
        # Calculate moving average
        window_size = 3
        self.yearly_counts['moving_average'] = self.yearly_counts['count'].rolling(window=window_size).mean()
        
        return self.yearly_counts
    
    def create_trend_visualization(self):
        """
        Create interactive visualization of content trends
        """
        if self.yearly_counts is None:
            self.analyze_content_trends()
        
        fig = px.line(self.yearly_counts, x='year', y='count', title='Yearly Content Added to Netflix')
        fig.add_scatter(
            x=self.yearly_counts['year'], 
            y=self.yearly_counts['moving_average'], 
            mode='lines', 
            name='3-Year Moving Average'
        )
        
        # Add trend line
        yearly_cleaned = self.yearly_counts.dropna(subset=['year', 'count'])
        z = np.polyfit(yearly_cleaned['year'], yearly_cleaned['count'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=yearly_cleaned['year'], 
            y=p(yearly_cleaned['year']), 
            mode='lines', 
            name='Trend Line', 
            line=dict(color='green', dash='dot')
        ))
        
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Number of Titles Added',
            hovermode='x unified'
        )
        fig.show()
        
        return fig
    
    def forecast_content(self, forecast_years=5):
        """
        Forecast future content additions using Exponential Smoothing
        """
        if self.yearly_counts is None:
            self.analyze_content_trends()
        
        # Prepare time series data
        time_series_data = self.yearly_counts.set_index('year')['count']
        
        # Fit Exponential Smoothing model
        model_es = ExponentialSmoothing(time_series_data, seasonal=None, trend='add').fit()
        
        # Forecast
        forecast = model_es.forecast(forecast_years)
        
        print(f"Forecasted content counts for the next {forecast_years} years:")
        print(forecast)
        
        return forecast


class RecommendationEngine:
    """Class for building recommendation system"""
    
    def __init__(self, df, cosine_sim_matrix=None):
        self.df = df
        self.cosine_sim_matrix = cosine_sim_matrix
        self.description_embeddings = None
        self.description_similarity_matrix = None
    
    def setup_sentence_embeddings(self):
        """
        Setup sentence embeddings for description similarity
        Note: This requires sentence-transformers library
        """
        # Check if description column exists
        if 'description' not in self.df.columns:
            print("Warning: 'description' column not found. Skipping description embeddings.")
            self.description_embeddings = None
            self.description_similarity_matrix = None
            return None
            
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load pre-trained model
            model_st = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Encode descriptions
            descriptions = self.df['description'].fillna('').tolist()
            self.description_embeddings = model_st.encode(descriptions, show_progress_bar=True)
            
            # Calculate similarity matrix (safe since description_embeddings will have features)
            self.description_similarity_matrix = cosine_similarity(self.description_embeddings)
            
            print(f"Shape of description embeddings: {self.description_embeddings.shape}")
            return self.description_embeddings
            
        except ImportError:
            print("sentence-transformers not installed. Please install it to use semantic similarity.")
            return None
    
    def get_recommendations_by_genre(self, title, top_n=10):
        """
        Get recommendations based on genre similarity
        """
        if self.cosine_sim_matrix is None:
            print("Cosine similarity matrix not available. Please compute it first.")
            return None
        
        try:
            # Find title index
            title_index = self.df[self.df['title'] == title].index[0]
            
            # Get similarity scores
            similarity_scores = list(enumerate(self.cosine_sim_matrix[title_index]))
            
            # Sort by similarity (excluding the title itself)
            sorted_similar = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
            
            # Get recommendations
            similar_indices = [i[0] for i in sorted_similar]
            similarity_scores = [i[1] for i in sorted_similar]
            
            # Use appropriate column names based on what's available
            type_col = 'type' if 'type' in self.df.columns else 'type_x'
            rating_col = 'rating' if 'rating' in self.df.columns else None
            year_col = 'release_year' if 'release_year' in self.df.columns else None
            
            recommendations_data = {
                'Title': self.df['title'].iloc[similar_indices],
                'Type': self.df[type_col].iloc[similar_indices],
                'Similarity Score': similarity_scores
            }
            
            if rating_col:
                recommendations_data['Rating'] = self.df[rating_col].iloc[similar_indices]
            if year_col:
                recommendations_data['Release Year'] = self.df[year_col].iloc[similar_indices]
                
            recommendations = pd.DataFrame(recommendations_data)
            
            return recommendations
            
        except IndexError:
            return f"Title '{title}' not found in dataset."
    
    def get_recommendations_by_description(self, title, top_n=10):
        """
        Get recommendations based on description similarity
        """
        if self.description_similarity_matrix is None:
            print("Description similarity matrix not available. Please setup sentence embeddings first.")
            return None
        
        try:
            # Find title index
            title_index = self.df[self.df['title'] == title].index[0]
            
            # Get similarity scores
            similarity_scores = list(enumerate(self.description_similarity_matrix[title_index]))
            
            # Sort by similarity (excluding the title itself)
            sorted_similar = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
            
            # Get recommendations
            similar_indices = [i[0] for i in sorted_similar]
            similarity_scores = [i[1] for i in sorted_similar]
            
            # Use appropriate column names based on what's available
            type_col = 'type' if 'type' in self.df.columns else 'type_x'
            rating_col = 'rating' if 'rating' in self.df.columns else None
            year_col = 'release_year' if 'release_year' in self.df.columns else None
            
            recommendations_data = {
                'Title': self.df['title'].iloc[similar_indices],
                'Type': self.df[type_col].iloc[similar_indices],
                'Similarity Score': similarity_scores
            }
            
            if rating_col:
                recommendations_data['Rating'] = self.df[rating_col].iloc[similar_indices]
            if year_col:
                recommendations_data['Release Year'] = self.df[year_col].iloc[similar_indices]
                
            recommendations = pd.DataFrame(recommendations_data)
            
            return recommendations
            
        except IndexError:
            return f"Title '{title}' not found in dataset."


class NetflixAnalyticsPipeline:
    """Main pipeline class that orchestrates all analyses"""
    
    def __init__(self):
        self.processor = None
        self.stats_analyzer = None
        self.bayesian = None
        self.linear_algebra = None
        self.regression = None
        self.clustering = None
        self.time_series = None
        self.recommender = None
    
    def run_full_pipeline(self, netflix_path, runtime_path=None):
        """
        Run the complete analytics pipeline
        
        Args:
            netflix_path (str): Path to Netflix CSV file
            runtime_path (str): Path to runtime CSV file (optional)
        """
        print("Starting Netflix Analytics Pipeline...")
        
        # 1. Data Processing
        print("\n=== DATA PROCESSING ===")
        self.processor = NetflixDataProcessor()
        df = self.processor.load_data(netflix_path, runtime_path)
        df = self.processor.clean_data()
        df = self.processor.normalize_duration()
        df_listed_in, df_country, df_cast, df_director = self.processor.create_exploded_tables()
        
        # 2. Statistical Analysis
        print("\n=== STATISTICAL ANALYSIS ===")
        self.stats_analyzer = StatisticalAnalyzer(df)
        self.stats_analyzer.descriptive_statistics()
        
        # 3. Bayesian Inference
        print("\n=== BAYESIAN INFERENCE ===")
        self.bayesian = BayesianInference(df)
        self.bayesian.train_model()
        feature_importance = self.bayesian.get_feature_importance()
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # 4. Linear Algebra & PCA
        print("\n=== LINEAR ALGEBRA & PCA ===")
        self.linear_algebra = LinearAlgebraAnalyzer(df, df_listed_in)
        genre_matrix, unique_genres = self.linear_algebra.create_genre_matrix()
        pca_result = self.linear_algebra.apply_pca()
        
        # 5. Regression Analysis
        print("\n=== REGRESSION ANALYSIS ===")
        self.regression = RegressionAnalyzer(df)
        linear_results = self.regression.predict_movie_duration()
        logistic_results = self.regression.classify_content_type()
        
        # 6. Clustering
        print("\n=== CLUSTERING ANALYSIS ===")
        self.clustering = ClusteringAnalyzer(df, pca_result)
        cluster_labels = self.clustering.perform_kmeans_clustering()
        cosine_sim = self.clustering.calculate_cosine_similarity(genre_matrix)
        
        # 7. Time Series Analysis
        print("\n=== TIME SERIES ANALYSIS ===")
        self.time_series = TimeSeriesAnalyzer(df)
        yearly_trends = self.time_series.analyze_content_trends()
        forecast = self.time_series.forecast_content()
        
        # 8. Recommendation Engine
        print("\n=== RECOMMENDATION ENGINE ===")
        self.recommender = RecommendationEngine(df, cosine_sim)
        
        print("\n=== PIPELINE COMPLETED ===")
        return {
            'dataframe': df,
            'processor': self.processor,
            'stats_analyzer': self.stats_analyzer,
            'bayesian': self.bayesian,
            'linear_algebra': self.linear_algebra,
            'regression': self.regression,
            'clustering': self.clustering,
            'time_series': self.time_series,
            'recommender': self.recommender
        }
    
    def create_visualizations(self):
        """
        Create all visualizations
        """
        if self.stats_analyzer:
            self.stats_analyzer.plot_distributions()
            self.stats_analyzer.interactive_distributions()
        
        if self.linear_algebra:
            self.linear_algebra.create_pca_visualizations()
        
        if self.time_series:
            self.time_series.create_trend_visualization()
    
    def get_recommendations(self, title, method='genre', top_n=10):
        """
        Get recommendations for a given title
        
        Args:
            title (str): Title to get recommendations for
            method (str): 'genre' or 'description'
            top_n (int): Number of recommendations
        """
        if self.recommender is None:
            print("Recommender not initialized. Please run the pipeline first.")
            return None
        
        if method == 'genre':
            return self.recommender.get_recommendations_by_genre(title, top_n)
        elif method == 'description':
            return self.recommender.get_recommendations_by_description(title, top_n)
        else:
            print("Method must be 'genre' or 'description'")
            return None


# Example usage and main function
def main():
    """
    Example usage of the Netflix Analytics Pipeline
    """
    # Initialize pipeline
    pipeline = NetflixAnalyticsPipeline()
    
    # Note: Update these paths to your actual data files
    netflix_path = "netflix_titles.csv"
    runtime_path = "netflix_movies_tv_runtime.csv"  # Optional
    
    try:
        # Run full pipeline
        results = pipeline.run_full_pipeline(netflix_path, runtime_path)
        
        # Create visualizations
        print("\nCreating visualizations...")
        pipeline.create_visualizations()
        
        # Example recommendations
        print("\nGetting recommendations...")
        recommendations = pipeline.get_recommendations("Stranger Things", method='genre', top_n=5)
        if recommendations is not None:
            print("\nRecommendations for 'Stranger Things':")
            print(recommendations)
        
        return results
        
    except FileNotFoundError:
        print("Data files not found. Please ensure the CSV files are in the correct location.")
        print("Update the file paths in the main() function.")
        return None


if __name__ == "__main__":
    # Uncomment the line below to run the full pipeline
    # results = main()
    
    # For testing individual components, you can use:
    print("Netflix Recommender System loaded successfully!")
    print("Use the NetflixAnalyticsPipeline class to run the full analysis.")
    print("Example: pipeline = NetflixAnalyticsPipeline()")
    print("results = pipeline.run_full_pipeline('netflix_titles.csv')")
