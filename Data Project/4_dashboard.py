"""
Netflix Interactive Dashboard Module
===================================

This module creates an interactive Streamlit dashboard for Netflix data exploration.
It provides user-friendly interfaces for data visualization and analysis.

Author: Netflix Analysis Project
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Netflix Content Analysis Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class NetflixDashboard:
    """
    Interactive Streamlit dashboard for Netflix data analysis
    """
    
    def __init__(self, data_path: str = 'data/netflix_cleaned.csv'):
        """
        Initialize the dashboard
        
        Args:
            data_path (str): Path to the cleaned Netflix CSV file
        """
        self.data_path = data_path
        self.df = None
        self.colors = ['#E50914', '#221F1F', '#F5F5F1', '#B81D24', '#831010']  # Netflix colors
        
    @st.cache_data
    def load_data(_self) -> pd.DataFrame:
        """Load the cleaned Netflix dataset with caching"""
        try:
            df = pd.read_csv(_self.data_path)
            return df
        except FileNotFoundError:
            st.error(f"❌ File not found: {_self.data_path}")
            st.info("💡 Please run 1_clean_data.py first to generate cleaned data")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return pd.DataFrame()
    
    def display_header(self):
        """Display dashboard header"""
        st.title("🎬 Netflix Content Analysis Dashboard")
        st.markdown("---")
        
        if not self.df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Titles", f"{len(self.df):,}")
            
            with col2:
                movies = len(self.df[self.df['type'] == 'Movie'])
                st.metric("Movies", f"{movies:,}")
            
            with col3:
                tv_shows = len(self.df[self.df['type'] == 'TV Show'])
                st.metric("TV Shows", f"{tv_shows:,}")
            
            with col4:
                countries = self.df['country'].nunique()
                st.metric("Countries", f"{countries:,}")
            
            st.markdown("---")
    
    def content_overview(self):
        """Display content overview section"""
        st.header("📊 Content Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Content type distribution pie chart
            type_counts = self.df['type'].value_counts()
            
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Content Type Distribution",
                color_discrete_sequence=[self.colors[0], self.colors[3]]
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Release year distribution
            fig = px.histogram(
                self.df,
                x='release_year',
                title="Content Distribution by Release Year",
                nbins=50,
                color_discrete_sequence=[self.colors[0]]
            )
            fig.update_layout(
                xaxis_title="Release Year",
                yaxis_title="Number of Titles",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def geographical_analysis(self):
        """Display geographical analysis section"""
        st.header("🌍 Geographical Analysis")
        
        # Process countries
        all_countries = []
        for countries_str in self.df['country'].dropna():
            if str(countries_str) not in ['Unknown Country', 'nan']:
                countries = [country.strip() for country in str(countries_str).split(',')]
                all_countries.extend(countries)
        
        country_counts = Counter(all_countries)
        
        # Remove unknown countries
        if 'Unknown Country' in country_counts:
            del country_counts['Unknown Country']
        
        # User controls
        col1, col2 = st.columns([1, 3])
        
        with col1:
            top_n_countries = st.slider("Number of top countries to display", 5, 30, 15)
        
        with col2:
            chart_type = st.selectbox("Chart type", ["Bar Chart", "Horizontal Bar Chart"])
        
        # Create chart
        top_countries = dict(country_counts.most_common(top_n_countries))
        
        if chart_type == "Bar Chart":
            fig = px.bar(
                x=list(top_countries.keys()),
                y=list(top_countries.values()),
                title=f"Top {top_n_countries} Countries by Content Count",
                color_discrete_sequence=[self.colors[0]]
            )
            fig.update_layout(
                xaxis_title="Country",
                yaxis_title="Number of Titles",
                xaxis_tickangle=-45
            )
        else:
            fig = px.bar(
                x=list(top_countries.values()),
                y=list(top_countries.keys()),
                orientation='h',
                title=f"Top {top_n_countries} Countries by Content Count",
                color_discrete_sequence=[self.colors[0]]
            )
            fig.update_layout(
                xaxis_title="Number of Titles",
                yaxis_title="Country"
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Country statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Countries", len(country_counts))
        
        with col2:
            us_content = country_counts.get('United States', 0)
            st.metric("US Content", f"{us_content:,}")
        
        with col3:
            international_content = sum(country_counts.values()) - us_content
            st.metric("International Content", f"{international_content:,}")
    
    def genre_analysis(self):
        """Display genre analysis section"""
        st.header("🎭 Genre Analysis")
        
        # Process genres
        all_genres = []
        movie_genres = []
        tv_genres = []
        
        for _, row in self.df.iterrows():
            if pd.notna(row['listed_in']) and str(row['listed_in']) not in ['Unknown Genre', 'nan']:
                genres = [genre.strip() for genre in str(row['listed_in']).split(',')]
                all_genres.extend(genres)
                
                if row['type'] == 'Movie':
                    movie_genres.extend(genres)
                else:
                    tv_genres.extend(genres)
        
        genre_counts = Counter(all_genres)
        movie_genre_counts = Counter(movie_genres)
        tv_genre_counts = Counter(tv_genres)
        
        # User controls
        col1, col2 = st.columns(2)
        
        with col1:
            top_n_genres = st.slider("Number of top genres to display", 5, 25, 15)
        
        with col2:
            analysis_type = st.selectbox("Analysis type", ["Overall", "Movies vs TV Shows", "Movies Only", "TV Shows Only"])
        
        # Create appropriate chart based on selection
        if analysis_type == "Overall":
            top_genres = dict(genre_counts.most_common(top_n_genres))
            fig = px.bar(
                x=list(top_genres.values()),
                y=list(top_genres.keys()),
                orientation='h',
                title=f"Top {top_n_genres} Genres Overall",
                color_discrete_sequence=[self.colors[0]]
            )
            
        elif analysis_type == "Movies vs TV Shows":
            # Create comparison chart
            top_movie_genres = dict(movie_genre_counts.most_common(top_n_genres))
            top_tv_genres = dict(tv_genre_counts.most_common(top_n_genres))
            
            # Get union of genres
            all_top_genres = set(list(top_movie_genres.keys()) + list(top_tv_genres.keys()))
            
            movie_counts = [movie_genre_counts.get(genre, 0) for genre in all_top_genres]
            tv_counts = [tv_genre_counts.get(genre, 0) for genre in all_top_genres]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Movies', y=list(all_top_genres), x=movie_counts, orientation='h', marker_color=self.colors[0]))
            fig.add_trace(go.Bar(name='TV Shows', y=list(all_top_genres), x=tv_counts, orientation='h', marker_color=self.colors[3]))
            
            fig.update_layout(
                title=f"Top {top_n_genres} Genres: Movies vs TV Shows",
                xaxis_title="Number of Titles",
                yaxis_title="Genre",
                barmode='group'
            )
            
        elif analysis_type == "Movies Only":
            top_genres = dict(movie_genre_counts.most_common(top_n_genres))
            fig = px.bar(
                x=list(top_genres.values()),
                y=list(top_genres.keys()),
                orientation='h',
                title=f"Top {top_n_genres} Movie Genres",
                color_discrete_sequence=[self.colors[0]]
            )
            
        else:  # TV Shows Only
            top_genres = dict(tv_genre_counts.most_common(top_n_genres))
            fig = px.bar(
                x=list(top_genres.values()),
                y=list(top_genres.keys()),
                orientation='h',
                title=f"Top {top_n_genres} TV Show Genres",
                color_discrete_sequence=[self.colors[3]]
            )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    def content_trends(self):
        """Display content trends over time"""
        st.header("📈 Content Trends Over Time")
        
        # User controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            year_range = st.slider(
                "Select year range",
                min_value=int(self.df['release_year'].min()),
                max_value=int(self.df['release_year'].max()),
                value=(1990, int(self.df['release_year'].max())),
                step=1
            )
        
        with col2:
            chart_type = st.selectbox("Chart type", ["Line Chart", "Area Chart", "Bar Chart"])
        
        with col3:
            show_separate = st.checkbox("Show Movies and TV Shows separately", value=True)
        
        # Filter data by year range
        filtered_df = self.df[
            (self.df['release_year'] >= year_range[0]) & 
            (self.df['release_year'] <= year_range[1])
        ]
        
        if show_separate:
            # Separate by content type
            yearly_content = filtered_df.groupby(['release_year', 'type']).size().unstack(fill_value=0)
            
            if chart_type == "Line Chart":
                fig = px.line(
                    yearly_content,
                    title="Content Release Trends Over Time",
                    color_discrete_sequence=[self.colors[0], self.colors[3]]
                )
            elif chart_type == "Area Chart":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=yearly_content.index,
                    y=yearly_content.get('Movie', []),
                    mode='lines',
                    name='Movies',
                    fill='tonexty',
                    line_color=self.colors[0]
                ))
                fig.add_trace(go.Scatter(
                    x=yearly_content.index,
                    y=yearly_content.get('TV Show', []),
                    mode='lines',
                    name='TV Shows',
                    fill='tozeroy',
                    line_color=self.colors[3]
                ))
                fig.update_layout(title="Content Release Trends Over Time")
            else:  # Bar Chart
                fig = px.bar(
                    yearly_content,
                    title="Content Release Trends Over Time",
                    color_discrete_sequence=[self.colors[0], self.colors[3]]
                )
        else:
            # Combined view
            yearly_total = filtered_df.groupby('release_year').size()
            
            if chart_type == "Line Chart":
                fig = px.line(
                    x=yearly_total.index,
                    y=yearly_total.values,
                    title="Total Content Release Trends Over Time",
                    color_discrete_sequence=[self.colors[0]]
                )
            elif chart_type == "Area Chart":
                fig = px.area(
                    x=yearly_total.index,
                    y=yearly_total.values,
                    title="Total Content Release Trends Over Time",
                    color_discrete_sequence=[self.colors[0]]
                )
            else:  # Bar Chart
                fig = px.bar(
                    x=yearly_total.index,
                    y=yearly_total.values,
                    title="Total Content Release Trends Over Time",
                    color_discrete_sequence=[self.colors[0]]
                )
        
        fig.update_layout(
            xaxis_title="Release Year",
            yaxis_title="Number of Titles",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def rating_analysis(self):
        """Display rating analysis section"""
        st.header("⭐ Content Rating Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution pie chart
            rating_counts = self.df['rating'].value_counts()
            
            fig = px.pie(
                values=rating_counts.values,
                names=rating_counts.index,
                title="Content Rating Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rating by content type
            rating_type = pd.crosstab(self.df['rating'], self.df['type'])
            
            fig = px.bar(
                rating_type,
                title="Content Rating by Type",
                color_discrete_sequence=[self.colors[0], self.colors[3]],
                barmode='group'
            )
            fig.update_layout(
                xaxis_title="Content Rating",
                yaxis_title="Number of Titles",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Age appropriateness analysis
        st.subheader("Age Appropriateness Categories")
        
        family_friendly = ['G', 'PG', 'TV-G', 'TV-Y', 'TV-Y7', 'TV-PG']
        teen_content = ['PG-13', 'TV-14']
        mature_content = ['R', 'TV-MA', 'NC-17']
        
        family_count = len(self.df[self.df['rating'].isin(family_friendly)])
        teen_count = len(self.df[self.df['rating'].isin(teen_content)])
        mature_count = len(self.df[self.df['rating'].isin(mature_content)])
        other_count = len(self.df) - family_count - teen_count - mature_count
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Family Friendly", f"{family_count:,}", f"{family_count/len(self.df)*100:.1f}%")
        
        with col2:
            st.metric("Teen Content", f"{teen_count:,}", f"{teen_count/len(self.df)*100:.1f}%")
        
        with col3:
            st.metric("Mature Content", f"{mature_count:,}", f"{mature_count/len(self.df)*100:.1f}%")
        
        with col4:
            st.metric("Other/Unrated", f"{other_count:,}", f"{other_count/len(self.df)*100:.1f}%")
    
    def content_search_and_filter(self):
        """Display content search and filter section"""
        st.header("🔍 Content Search & Filter")
        
        # Search and filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            content_type = st.selectbox("Content Type", ["All", "Movie", "TV Show"])
        
        with col2:
            countries = ["All"] + sorted(self.df['country'].dropna().unique().tolist())
            selected_country = st.selectbox("Country", countries)
        
        with col3:
            # Get all unique genres
            all_genres = set()
            for genres_str in self.df['listed_in'].dropna():
                if str(genres_str) not in ['Unknown Genre', 'nan']:
                    genres = [genre.strip() for genre in str(genres_str).split(',')]
                    all_genres.update(genres)
            
            genres = ["All"] + sorted(list(all_genres))
            selected_genre = st.selectbox("Genre", genres)
        
        # Text search
        search_term = st.text_input("Search titles, directors, or cast", "")
        
        # Apply filters
        filtered_df = self.df.copy()
        
        if content_type != "All":
            filtered_df = filtered_df[filtered_df['type'] == content_type]
        
        if selected_country != "All":
            filtered_df = filtered_df[filtered_df['country'].str.contains(selected_country, na=False)]
        
        if selected_genre != "All":
            filtered_df = filtered_df[filtered_df['listed_in'].str.contains(selected_genre, na=False)]
        
        if search_term:
            mask = (
                filtered_df['title'].str.contains(search_term, case=False, na=False) |
                filtered_df['director'].str.contains(search_term, case=False, na=False) |
                filtered_df['cast'].str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[mask]
        
        # Display results
        st.subheader(f"Found {len(filtered_df):,} titles")
        
        if len(filtered_df) > 0:
            # Display sample results
            display_df = filtered_df[['title', 'type', 'release_year', 'rating', 'duration', 'listed_in']].head(100)
            st.dataframe(display_df, use_container_width=True)
            
            # Download option
            if st.button("Download filtered results as CSV"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Click to download",
                    data=csv,
                    file_name=f"netflix_filtered_results.csv",
                    mime="text/csv"
                )
        else:
            st.info("No titles match your search criteria. Try adjusting your filters.")
    
    def run_dashboard(self):
        """Run the complete dashboard"""
        # Load data
        self.df = self.load_data()
        
        if self.df.empty:
            st.stop()
        
        # Sidebar navigation
        st.sidebar.title("🎬 Navigation")
        
        sections = [
            "📊 Overview",
            "🌍 Geographical Analysis",
            "🎭 Genre Analysis", 
            "📈 Content Trends",
            "⭐ Rating Analysis",
            "🔍 Search & Filter"
        ]
        
        selected_section = st.sidebar.radio("Select Section", sections)
        
        # Display header
        self.display_header()
        
        # Display selected section
        if selected_section == "📊 Overview":
            self.content_overview()
            
        elif selected_section == "🌍 Geographical Analysis":
            self.geographical_analysis()
            
        elif selected_section == "🎭 Genre Analysis":
            self.genre_analysis()
            
        elif selected_section == "📈 Content Trends":
            self.content_trends()
            
        elif selected_section == "⭐ Rating Analysis":
            self.rating_analysis()
            
        elif selected_section == "🔍 Search & Filter":
            self.content_search_and_filter()
        
        # Sidebar info
        st.sidebar.markdown("---")
        st.sidebar.info(
            """
            🎬 **Netflix Analysis Dashboard**
            
            This dashboard provides comprehensive analysis of Netflix content including:
            - Content distribution and trends
            - Geographical analysis
            - Genre popularity
            - Rating patterns
            - Interactive search and filtering
            
            **Data Source:** Netflix Titles Dataset
            **Last Updated:** October 2025
            """
        )


def main():
    """Main function to run the dashboard"""
    dashboard = NetflixDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()