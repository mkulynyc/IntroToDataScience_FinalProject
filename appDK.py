"""
Netflix Recommender System - Streamlit App
Interactive web application for Netflix content analysis and recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our recommender system
from recomender_v1 import NetflixAnalyticsPipeline

# Configure Streamlit page
st.set_page_config(
    page_title="Netflix Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants for file paths
PICKLE_DIR = "pickle_data"
PIPELINE_PICKLE = os.path.join(PICKLE_DIR, "netflix_pipeline.pkl")
DATA_PICKLE = os.path.join(PICKLE_DIR, "netflix_data.pkl")

class StreamlitNetflixApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.pipeline = None
        self.df = None
        self.results = None
        self.required_columns = {
            'type',
            'movie_minutes',
            'content_minutes',
            'release_year',
            'date_added',
            'rating',
            'country',
            'listed_in'
        }
        
        # Create pickle directory if it doesn't exist
        if not os.path.exists(PICKLE_DIR):
            os.makedirs(PICKLE_DIR)
    
    def validate_dataframe(self, df):
        """Ensure the dataframe contains the columns needed for charts."""
        if not isinstance(df, pd.DataFrame):
            return False
        
        missing_cols = self.required_columns - set(df.columns)
        if missing_cols:
            st.info(
                "Cached data is missing new runtime columns "
                f"({', '.join(sorted(missing_cols))}). Re-processing the dataset..."
            )
            return False
        return True
    
    def save_data_to_pickle(self, data, filename):
        """Save data to pickle file"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            st.error(f"Error saving data: {str(e)}")
            return False
    
    def load_data_from_pickle(self, filename):
        """Load data from pickle file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                return data
            return None
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def initialize_pipeline(self, netflix_path, runtime_path=None, force_reload=False):
        """Initialize or load the Netflix analytics pipeline"""
        
        if not force_reload:
            # Try to load existing pipeline from pickle
            saved_pipeline = self.load_data_from_pickle(PIPELINE_PICKLE)
            saved_data = self.load_data_from_pickle(DATA_PICKLE)
            
            # Check if both pipeline and data are valid
            pipeline_valid = saved_pipeline is not None
            data_valid = (
                saved_data is not None
                and (not hasattr(saved_data, 'empty') or not saved_data.empty)
                and self.validate_dataframe(saved_data)
            )
            
            if pipeline_valid and data_valid:
                st.success("Loaded existing processed data from pickle files!")
                self.pipeline = saved_pipeline
                self.df = saved_data
                return True
        
        # Process data if not loaded from pickle or force reload
        if netflix_path and os.path.exists(netflix_path):
            with st.spinner("Processing Netflix data... This may take a few minutes."):
                try:
                    self.pipeline = NetflixAnalyticsPipeline()
                    self.results = self.pipeline.run_full_pipeline(netflix_path, runtime_path)
                    self.df = self.results['dataframe']
                    
                    # Save to pickle for future use
                    self.save_data_to_pickle(self.pipeline, PIPELINE_PICKLE)
                    self.save_data_to_pickle(self.df, DATA_PICKLE)
                    
                    st.success("Data processed successfully and saved to pickle files!")
                    return True
                    
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    return False
        else:
            st.error("Netflix data file not found!")
            return False
    
    def render_sidebar(self):
        """Render the sidebar with data upload and settings"""
        st.sidebar.title("🎬 Netflix Recommender")
        st.sidebar.markdown("---")
        
        # File upload section
        st.sidebar.subheader("📁 Data Files")
        
        # Check for existing data files
        netflix_file = "netflix_titles.csv"
        runtime_file = "netflix_movies_tv_runtime.csv"
        
        netflix_exists = os.path.exists(netflix_file)
        runtime_exists = os.path.exists(runtime_file)
        
        if netflix_exists:
            st.sidebar.success(f"✅ Found: {netflix_file}")
        else:
            st.sidebar.error(f"❌ Missing: {netflix_file}")
        
        if runtime_exists:
            st.sidebar.info(f"ℹ️ Optional: {runtime_file}")
        else:
            st.sidebar.warning(f"⚠️ Optional file missing: {runtime_file}")
        
        # File upload widgets
        uploaded_netflix = st.sidebar.file_uploader(
            "Upload Netflix Titles CSV", 
            type=['csv'],
            help="Main Netflix dataset (netflix_titles.csv)"
        )
        
        uploaded_runtime = st.sidebar.file_uploader(
            "Upload Runtime Data CSV (Optional)", 
            type=['csv'],
            help="Additional runtime information (netflix_movies_tv_runtime.csv)"
        )
        
        # Save uploaded files
        if uploaded_netflix:
            with open(netflix_file, "wb") as f:
                f.write(uploaded_netflix.getbuffer())
            st.sidebar.success("Netflix file uploaded!")
            netflix_exists = True
        
        if uploaded_runtime:
            with open(runtime_file, "wb") as f:
                f.write(uploaded_runtime.getbuffer())
            st.sidebar.success("Runtime file uploaded!")
            runtime_exists = True
        
        st.sidebar.markdown("---")
        
        # Control buttons
        st.sidebar.subheader("🔧 Controls")
        
        process_button = st.sidebar.button(
            "🚀 Process Data", 
            disabled=not netflix_exists,
            help="Process the Netflix data and build models"
        )
        
        reload_button = st.sidebar.button(
            "🔄 Force Reload", 
            help="Force reload data even if pickle files exist"
        )
        
        clear_cache_button = st.sidebar.button(
            "🗑️ Clear Cache",
            help="Clear all pickle files and start fresh"
        )
        
        if clear_cache_button:
            self.clear_cache()
        
        return netflix_exists, runtime_exists, process_button, reload_button
    
    def clear_cache(self):
        """Clear all pickle files"""
        try:
            if os.path.exists(PIPELINE_PICKLE):
                os.remove(PIPELINE_PICKLE)
            if os.path.exists(DATA_PICKLE):
                os.remove(DATA_PICKLE)
            st.sidebar.success("Cache cleared!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error clearing cache: {str(e)}")
    
    def render_main_content(self):
        """Render the main content area"""
        
        if self.df is None:
            st.title("🎬 Netflix Recommender System")
            st.markdown("""
            Welcome to the Netflix Content Analysis and Recommendation System!
            
            ### Getting Started:
            1. **Upload Data**: Use the sidebar to upload your Netflix dataset
            2. **Process Data**: Click "Process Data" to run the analytics pipeline
            3. **Explore**: Navigate through different sections to explore insights
            4. **Get Recommendations**: Use the recommendation engine to find similar content
            
            ### Features:
            - 📊 **Statistical Analysis**: Comprehensive data analysis and visualizations
            - 🤖 **Machine Learning**: Bayesian inference, clustering, and regression models
            - 🎯 **Recommendations**: Content-based recommendation system
            - 📈 **Time Series**: Trend analysis and forecasting
            - 🎨 **Interactive Visualizations**: Plotly-powered charts and graphs
            """)
            
            # Show sample data info
            st.subheader("📋 Expected Data Format")
            sample_data = {
                'show_id': ['s1', 's2', 's3'],
                'type': ['Movie', 'TV Show', 'Movie'],
                'title': ['Example Movie 1', 'Example Series 1', 'Example Movie 2'],
                'rating': ['PG-13', 'TV-MA', 'R'],
                'release_year': [2020, 2019, 2021],
                'duration': ['120 min', '3 Seasons', '95 min'],
                'listed_in': ['Drama, Thriller', 'Sci-Fi, Horror', 'Comedy, Romance']
            }
            st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
            
            return
        
        # Main navigation
        st.title("🎬 Netflix Analytics Dashboard")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", 
            "🎯 Recommendations", 
            "📈 Statistics", 
            "🔍 Analysis", 
            "⏰ Time Series",
            "🎨 Visualizations"
        ])
        
        with tab1:
            self.render_overview_tab()
        
        with tab2:
            self.render_recommendations_tab()
        
        with tab3:
            self.render_statistics_tab()
        
        with tab4:
            self.render_analysis_tab()
        
        with tab5:
            self.render_timeseries_tab()
        
        with tab6:
            self.render_visualizations_tab()
    
    def render_overview_tab(self):
        """Render the overview tab"""
        st.subheader("📊 Dataset Overview")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Titles", len(self.df))
        
        with col2:
            if 'type' in self.df.columns:
                movies_count = len(self.df[self.df['type'] == 'Movie'])
            else:
                movies_count = 0
            st.metric("Movies", movies_count)
        
        with col3:
            if 'type' in self.df.columns:
                tv_count = len(self.df[self.df['type'] == 'TV Show'])
            else:
                tv_count = 0
            st.metric("TV Shows", tv_count)
        
        with col4:
            if 'country' in self.df.columns:
                unique_countries = self.df['country'].apply(
                    lambda x: len(x) if isinstance(x, list) else 1
                ).sum()
                st.metric("Countries", unique_countries)
            else:
                st.metric("Countries", "N/A")
        
        # Content type distribution
        st.subheader("📺 Content Distribution")
        if 'type' in self.df.columns:
            type_counts = self.df['type'].value_counts()
            fig_pie = px.pie(
                values=type_counts.values, 
                names=type_counts.index,
                title="Movies vs TV Shows Distribution"
            )
            st.plotly_chart(fig_pie, width='stretch')
        else:
            st.warning("Type information not available for content distribution chart")
        
        # Recent additions
        st.subheader("📅 Recent Additions")
        if 'date_added' in self.df.columns:
            recent_data = self.df.dropna(subset=['date_added']).nlargest(10, 'date_added')
            
            # Build column list based on availability
            display_cols = ['title']
            if 'type' in self.df.columns:
                display_cols.append('type')
            
            display_cols.append('date_added')
            
            if 'rating' in self.df.columns:
                display_cols.append('rating')
            if 'release_year' in self.df.columns:
                display_cols.append('release_year')
            
            st.dataframe(
                recent_data[display_cols],
                use_container_width=True
            )
    
    def render_recommendations_tab(self):
        """Render the recommendations tab"""
        st.subheader("🎯 Get Recommendations")
        
        if not hasattr(self.pipeline, 'recommender') or self.pipeline.recommender is None:
            st.warning("Recommender not initialized. Please process the data first.")
            return
        
        # Title selection
        available_titles = self.df['title'].dropna().unique()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_title = st.selectbox(
                "Select a title to get recommendations:",
                options=available_titles,
                index=0 if len(available_titles) > 0 else None
            )
        
        with col2:
            recommendation_method = st.selectbox(
                "Recommendation Method:",
                options=['genre', 'description'],
                help="Genre: Based on similar genres\nDescription: Based on content similarity"
            )
        
        num_recommendations = st.slider("Number of recommendations:", 1, 20, 10)
        
        if st.button("🔍 Get Recommendations"):
            if selected_title:
                with st.spinner("Finding similar content..."):
                    try:
                        recommendations = self.pipeline.get_recommendations(
                            selected_title, 
                            method=recommendation_method, 
                            top_n=num_recommendations
                        )
                        
                        if recommendations is not None and not recommendations.empty:
                            st.success(f"Found {len(recommendations)} recommendations!")
                            
                            # Display recommendations
                            st.subheader(f"🎬 Recommendations for '{selected_title}'")
                            
                            # Format the recommendations table
                            formatted_recs = recommendations.copy()
                            if 'Similarity Score' in formatted_recs.columns:
                                formatted_recs['Similarity Score'] = formatted_recs['Similarity Score'].round(3)
                            
                            st.dataframe(formatted_recs, use_container_width=True)
                            
                            # Show recommendation distribution
                            if 'Type' in recommendations.columns:
                                type_dist = recommendations['Type'].value_counts()
                                if len(type_dist) > 1:
                                    fig_rec_dist = px.bar(
                                        x=type_dist.index, 
                                        y=type_dist.values,
                                        title="Recommendation Type Distribution",
                                        labels={'x': 'Content Type', 'y': 'Count'}
                                    )
                                    st.plotly_chart(fig_rec_dist, width='stretch')
                                else:
                                    only_type = type_dist.index[0]
                                    st.info(f"All recommendations are {only_type}s for this title.")
                        
                        else:
                            st.error("No recommendations found for this title.")
                    
                    except Exception as e:
                        st.error(f"Error getting recommendations: {str(e)}")
            else:
                st.warning("Please select a title first.")
    
    def render_statistics_tab(self):
        """Render the statistics tab"""
        st.subheader("📈 Statistical Analysis")
        
        # Descriptive statistics
        duration_series = None
        duration_label = "Movie Duration"
        if 'movie_minutes' in self.df.columns and self.df['movie_minutes'].dropna().any():
            duration_series = self.df.loc[self.df['type'] == 'Movie', 'movie_minutes']
        if (duration_series is None or duration_series.dropna().empty) and 'content_minutes' in self.df.columns:
            duration_series = self.df['content_minutes']
            duration_label = "Content Duration"
        
        if duration_series is not None and not duration_series.dropna().empty:
            st.subheader(f"🎬 {duration_label} Statistics")
            clean_duration = duration_series.dropna()
            stats = clean_duration.describe()
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(stats.round(2))
            
            with col2:
                fig_hist = px.histogram(
                    x=clean_duration,
                    nbins=40,
                    title=f"{duration_label} Distribution",
                    labels={'x': f'{duration_label} (minutes)', 'y': 'Count'},
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.update_traces(marker=dict(line=dict(color='white', width=1)))
                fig_hist.update_layout(
                    showlegend=False,
                    xaxis_title=f"{duration_label} (minutes)",
                    yaxis_title="Number of Titles"
                )
                st.plotly_chart(fig_hist, width='stretch')
        
        # Release year statistics
        if 'release_year' in self.df.columns:
            st.subheader("📅 Release Year Statistics")
            release_year = pd.to_numeric(self.df['release_year'], errors='coerce').dropna()
            if not release_year.empty:
                year_stats = release_year.describe()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(year_stats.round(0))
                
                with col2:
                    fig_year_hist = px.histogram(
                        release_year,
                        nbins=20,
                        title="Release Year Distribution"
                    )
                    fig_year_hist.update_traces(marker=dict(line=dict(color='white', width=1)))
                    st.plotly_chart(fig_year_hist, width='stretch')
            else:
                st.info("Release year data not available after processing.")
            
        # Rating distribution
        # Check for TMDB rating from runtime CSV (prioritize _runtime suffix to get numeric TMDB ratings)
        rating_col = None
        
        # First check all columns to find any numeric rating column
        for col in self.df.columns:
            if 'rating' in col.lower() and '_runtime' in col.lower():
                # Try to check if it's numeric
                try:
                    test_vals = pd.to_numeric(self.df[col], errors='coerce').dropna()
                    if len(test_vals) > 0 and test_vals.max() <= 10:  # TMDB ratings are 0-10
                        rating_col = col
                        break
                except:
                    continue
        
        # Fallback to checking specific column names
        if not rating_col:
            possible_rating_cols = ['rating_stars_runtime', 'rating_runtime', 'total_rating_runtime', 'total_rating', 'vote_average', 'tmdb_rating']
            for col in possible_rating_cols:
                if col in self.df.columns:
                    rating_col = col
                    break
            
        if rating_col:
            try:
                st.subheader("⭐ TMDB Rating Distribution")
                
                # Get ratings, convert to numeric, and clean data
                ratings = pd.to_numeric(self.df[rating_col], errors='coerce').dropna()
                
                if len(ratings) > 0:
                    # Create histogram with rating bins
                    fig_rating = px.histogram(
                        x=ratings,
                        nbins=20,
                        title="TMDB Rating Distribution (0-10 scale)",
                        labels={'x': 'Rating (0-10)', 'y': 'Number of Titles'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_rating.update_traces(marker=dict(line=dict(color='white', width=1)))
                    fig_rating.update_layout(
                        xaxis_title="Rating (0-10)",
                        yaxis_title="Number of Titles",
                        showlegend=False
                    )
                    st.plotly_chart(fig_rating, width='stretch')
                    
                    # Show rating statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Rating", f"{ratings.mean():.2f}/10")
                    with col2:
                        st.metric("Median Rating", f"{ratings.median():.2f}/10")
                    with col3:
                        st.metric("Total Rated", f"{len(ratings):,}")
                else:
                    st.info(f"No valid rating data found in column '{rating_col}'.")
            except Exception as e:
                st.error(f"Error displaying rating distribution: {str(e)}")
        else:
            # Debug: Show what columns are available
            with st.expander("🔍 Debug: Available columns"):
                st.write("Looking for TMDB rating columns. All columns with 'rating':")
                rating_cols = [col for col in self.df.columns if 'rating' in col.lower()]
                st.write(rating_cols)
                st.write("\nAll columns:")
                st.write(list(self.df.columns))
            st.info("No TMDB rating column found. Please ensure netflix_movies_tv_runtime.csv is loaded correctly.")
    
    def render_analysis_tab(self):
        """Render the analysis tab"""
        st.subheader("🔍 Advanced Analysis")
        
        # Model results if available
        if hasattr(self.pipeline, 'bayesian') and self.pipeline.bayesian:
            st.subheader("🤖 Bayesian Classification Results")
            
            try:
                feature_importance = self.pipeline.bayesian.get_feature_importance(top_n=10)
                st.dataframe(feature_importance.round(4), use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying Bayesian results: {str(e)}")
        
        # Clustering results if available
        if 'cluster_label' in self.df.columns:
            st.subheader("🎯 Cluster Analysis")
            
            cluster_counts = self.df['cluster_label'].value_counts().sort_index()
            fig_clusters = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                title="Content Clusters Distribution"
            )
            st.plotly_chart(fig_clusters, width='stretch')
            
            # Show sample from each cluster
            st.subheader("📋 Sample Content by Cluster")
            selected_cluster = st.selectbox(
                "Select cluster to view sample content:",
                options=sorted(self.df['cluster_label'].unique())
            )
            
            cluster_sample = self.df[self.df['cluster_label'] == selected_cluster].head(10)
            
            # Build column list based on availability
            display_cols = ['title']
            if 'type' in cluster_sample.columns:
                display_cols.append('type')
            elif 'type' in cluster_sample.columns:
                display_cols.append('type')
            
            if 'rating' in cluster_sample.columns:
                display_cols.append('rating')
            if 'release_year' in cluster_sample.columns:
                display_cols.append('release_year')
            
            st.dataframe(
                cluster_sample[display_cols],
                use_container_width=True
            )
    
    def render_timeseries_tab(self):
        """Render the time series tab"""
        st.subheader("⏰ Time Series Analysis")
        
        date_field_available = (
            'date_added' in self.df.columns and
            pd.to_datetime(self.df['date_added'], errors='coerce').notna().any()
        )
        release_year_available = (
            'release_year' in self.df.columns and
            pd.to_numeric(self.df['release_year'], errors='coerce').notna().any()
        )
        
        if date_field_available:
            df_with_year = self.df.copy()
            df_with_year['date_added'] = pd.to_datetime(df_with_year['date_added'], errors='coerce')
            df_with_year['year_added'] = df_with_year['date_added'].dt.year
            df_with_year['month_added'] = df_with_year['date_added'].dt.month
            
            yearly_counts = df_with_year.groupby('year_added').size()
            fig_trend = px.line(
                x=yearly_counts.index,
                y=yearly_counts.values,
                title="Netflix Content Additions Over Time",
                labels={'x': 'Year', 'y': 'Number of Titles Added'}
            )
            st.plotly_chart(fig_trend, width='stretch')
            
            monthly_counts = df_with_year['month_added'].value_counts().sort_index()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            labeled_months = [month_names[int(i) - 1] for i in monthly_counts.index if 1 <= int(i) <= 12]
            fig_monthly = px.bar(
                x=labeled_months,
                y=monthly_counts.values,
                title="Content Additions by Month",
                labels={'x': 'Month', 'y': 'Number of Titles'},
                color_discrete_sequence=['#1f77b4']
            )
            fig_monthly.update_traces(marker=dict(line=dict(color='white', width=1)))
            fig_monthly.update_layout(
                xaxis_title="Month",
                yaxis_title="Number of Titles"
            )
            st.plotly_chart(fig_monthly, width='stretch')
        elif release_year_available:
            release_years = pd.to_numeric(self.df['release_year'], errors='coerce').dropna().astype(int)
            year_counts = release_years.value_counts().sort_index()
            fig_trend = px.line(
                x=year_counts.index,
                y=year_counts.values,
                title="Netflix Content Releases Over Time",
                labels={'x': 'Year', 'y': 'Number of Titles'}
            )
            st.plotly_chart(fig_trend, width='stretch')
        else:
            st.warning("Date or release year information not available for time series analysis.")
    
    def render_visualizations_tab(self):
        """Render the visualizations tab"""
        st.subheader("🎨 Interactive Visualizations")
        
        # Genre analysis
        if hasattr(self, 'df') and self.df is not None:
            st.subheader("🎭 Genre Analysis")
            
            if 'primary_genre' in self.df.columns:
                genre_counts = self.df['primary_genre'].value_counts().head(15)
                fig_genres = px.bar(
                    x=genre_counts.values,
                    y=genre_counts.index,
                    orientation='h',
                    title="Top Genres / Runtime Buckets"
                )
                st.plotly_chart(fig_genres, width='stretch')
                
                if 'content_minutes' in self.df.columns:
                    genre_runtime = (
                        self.df[['primary_genre', 'content_minutes']]
                        .dropna()
                        .groupby('primary_genre')['content_minutes']
                        .mean()
                        .sort_values(ascending=False)
                        .head(15)
                    )
                    if not genre_runtime.empty:
                        fig_runtime = px.bar(
                            x=genre_runtime.values,
                            y=genre_runtime.index,
                            orientation='h',
                            title="Average Content Minutes by Genre"
                        )
                        st.plotly_chart(fig_runtime, width='stretch')
            else:
                st.info("Genre metadata unavailable; showing runtime and region trends instead.")
            
            # Country distribution
            if 'country' in self.df.columns:
                all_countries = []
                for countries in self.df['country'].dropna():
                    if isinstance(countries, list):
                        all_countries.extend([c for c in countries if c])
                    elif isinstance(countries, str):
                        all_countries.extend([c.strip() for c in countries.split(',') if c])
                
                if all_countries:
                    country_series = pd.Series(all_countries, dtype="object").astype(str).str.strip()
                    exclusions = {'', 'unknown', 'nan', 'none', 'n/a'}
                    country_series = country_series[~country_series.str.casefold().isin(exclusions)]
                    if not country_series.empty:
                        country_counts = country_series.value_counts().head(15)
                        fig_countries = px.bar(
                            x=country_counts.values,
                            y=country_counts.index,
                            orientation='h',
                            title="Top 15 Countries by Content Count"
                        )
                        st.plotly_chart(fig_countries, width='stretch')
                    else:
                        st.info("Country information not available for visualization.")
                else:
                    st.info("Country information not available for visualization.")
            
            # Release year vs Rating
            if 'release_year' in self.df.columns and 'rating' in self.df.columns:
                trend_df = self.df.copy()
                trend_df['release_year'] = pd.to_numeric(trend_df['release_year'], errors='coerce')
                trend_df = trend_df.dropna(subset=['release_year', 'rating'])
                
                if not trend_df.empty:
                    trend_df['release_year'] = trend_df['release_year'].astype(int)
                    rating_counts = (
                        trend_df
                        .groupby(['release_year', 'rating'])
                        .size()
                        .reset_index(name='count')
                    )
                    rating_counts['rating'] = rating_counts['rating'].astype(str)
                    rating_counts = rating_counts[
                        ~rating_counts['rating'].str.casefold().isin({'unknown', 'nan', 'none', ''})
                    ]
                    if not rating_counts.empty:
                        rating_counts = rating_counts.sort_values(['release_year', 'rating'])
                        fig_rating_trend = px.line(
                            rating_counts,
                            x='release_year',
                            y='count',
                            color='rating',
                            markers=True,
                            title="Content Rating Trend by Release Year",
                            labels={'count': 'Number of Titles', 'release_year': 'Release Year'}
                        )
                        st.plotly_chart(fig_rating_trend, width='stretch')
                    else:
                        st.info("Rating trend chart skipped: no rating data available.")
                else:
                    st.info("Insufficient release year data for rating trends.")
    
    def run(self):
        """Main application runner"""
        # Render sidebar
        netflix_exists, runtime_exists, process_button, reload_button = self.render_sidebar()
        
        # Initialize pipeline if button clicked or data exists
        if process_button or reload_button:
            netflix_path = "netflix_titles.csv" if netflix_exists else None
            runtime_path = "netflix_movies_tv_runtime.csv" if runtime_exists else None
            
            if self.initialize_pipeline(netflix_path, runtime_path, force_reload=reload_button):
                st.rerun()
        
        # Try to load existing data on startup
        if self.df is None:
            netflix_path = "netflix_titles.csv" if netflix_exists else None
            runtime_path = "netflix_movies_tv_runtime.csv" if runtime_exists else None
            self.initialize_pipeline(netflix_path, runtime_path, force_reload=False)
        
        # Render main content
        self.render_main_content()


def main():
    """Main function to run the Streamlit app"""
    app = StreamlitNetflixApp()
    app.run()


if __name__ == "__main__":
    main()
