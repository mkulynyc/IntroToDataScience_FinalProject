"""
Netflix Data Visualization Module
=================================

This module creates comprehensive static visualizations for the Netflix dataset.
It generates charts, plots, and graphs using matplotlib and seaborn.

Author: Netflix Analysis Project
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class NetflixVisualizer:
    """
    Comprehensive visualization generator for Netflix dataset
    """
    
    def __init__(self, data_path: str = 'data/netflix_cleaned.csv'):
        """
        Initialize the visualizer
        
        Args:
            data_path (str): Path to the cleaned Netflix CSV file
        """
        self.data_path = data_path
        self.df = None
        self.fig_size = (12, 8)
        self.colors = ['#E50914', '#221F1F', '#F5F5F1', '#B81D24', '#831010']  # Netflix colors
        
    def load_data(self) -> pd.DataFrame:
        """Load the cleaned Netflix dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Data loaded successfully: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
            return self.df
        except FileNotFoundError:
            print(f"❌ File not found: {self.data_path}")
            print("💡 Please run 1_clean_data.py first to generate cleaned data")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def plot_content_type_distribution(self, save_path: str = 'visualizations/content_type_distribution.png'):
        """Create pie chart showing Movies vs TV Shows distribution"""
        if self.df is None:
            return
        
        type_counts = self.df['type'].value_counts()
        
        plt.figure(figsize=(10, 8))
        colors = [self.colors[0], self.colors[3]]
        wedges, texts, autotexts = plt.pie(type_counts.values, 
                                          labels=type_counts.index, 
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          explode=(0.05, 0.05),
                                          shadow=True,
                                          startangle=90)
        
        plt.title('Netflix Content Distribution: Movies vs TV Shows', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        plt.tight_layout()
        self._save_plot(save_path)
        plt.show()
    
    def plot_content_trends_over_time(self, save_path: str = 'visualizations/content_trends_over_time.png'):
        """Create line chart showing content trends over years"""
        if self.df is None:
            return
        
        # Group by year and type
        yearly_content = self.df.groupby(['release_year', 'type']).size().unstack(fill_value=0)
        
        plt.figure(figsize=self.fig_size)
        
        # Plot lines for Movies and TV Shows
        plt.plot(yearly_content.index, yearly_content['Movie'], 
                color=self.colors[0], linewidth=3, marker='o', label='Movies', markersize=4)
        plt.plot(yearly_content.index, yearly_content['TV Show'], 
                color=self.colors[3], linewidth=3, marker='s', label='TV Shows', markersize=4)
        
        plt.title('Netflix Content Release Trends Over Time', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Release Year', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Titles', fontsize=12, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Focus on relevant years (remove outliers)
        plt.xlim(1940, 2025)
        
        plt.tight_layout()
        self._save_plot(save_path)
        plt.show()
    
    def plot_top_countries(self, top_n: int = 15, save_path: str = 'visualizations/top_countries.png'):
        """Create horizontal bar chart of top countries by content count"""
        if self.df is None:
            return
        
        # Process countries (handle multiple countries per title)
        all_countries = []
        for countries_str in self.df['country'].dropna():
            if str(countries_str) not in ['Unknown Country', 'nan']:
                countries = [country.strip() for country in str(countries_str).split(',')]
                all_countries.extend(countries)
        
        country_counts = Counter(all_countries)
        top_countries = dict(country_counts.most_common(top_n))
        
        plt.figure(figsize=(12, 10))
        countries = list(top_countries.keys())
        counts = list(top_countries.values())
        
        # Create horizontal bar chart
        bars = plt.barh(range(len(countries)), counts, color=self.colors[0])
        
        # Customize the plot
        plt.yticks(range(len(countries)), countries)
        plt.xlabel('Number of Titles', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Countries by Netflix Content Count', fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(count + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{count:,}', va='center', fontweight='bold')
        
        plt.gca().invert_yaxis()  # Highest at top
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        self._save_plot(save_path)
        plt.show()
    
    def plot_genre_distribution(self, top_n: int = 15, save_path: str = 'visualizations/genre_distribution.png'):
        """Create bar chart of top genres"""
        if self.df is None:
            return
        
        # Process genres (handle multiple genres per title)
        all_genres = []
        for genres_str in self.df['listed_in'].dropna():
            if str(genres_str) not in ['Unknown Genre', 'nan']:
                genres = [genre.strip() for genre in str(genres_str).split(',')]
                all_genres.extend(genres)
        
        genre_counts = Counter(all_genres)
        top_genres = dict(genre_counts.most_common(top_n))
        
        plt.figure(figsize=self.fig_size)
        genres = list(top_genres.keys())
        counts = list(top_genres.values())
        
        bars = plt.bar(range(len(genres)), counts, color=self.colors[0])
        
        plt.xticks(range(len(genres)), genres, rotation=45, ha='right')
        plt.ylabel('Number of Titles', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Genres on Netflix', fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save_plot(save_path)
        plt.show()
    
    def plot_rating_distribution(self, save_path: str = 'visualizations/rating_distribution.png'):
        """Create stacked bar chart of content ratings by type"""
        if self.df is None:
            return
        
        # Create crosstab
        rating_type = pd.crosstab(self.df['rating'], self.df['type'])
        
        plt.figure(figsize=self.fig_size)
        
        # Create stacked bar chart
        rating_type.plot(kind='bar', stacked=True, color=[self.colors[0], self.colors[3]], 
                        figsize=self.fig_size, ax=plt.gca())
        
        plt.title('Content Rating Distribution by Type', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Content Rating', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Titles', fontsize=12, fontweight='bold')
        plt.legend(title='Content Type', fontsize=10)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_plot(save_path)
        plt.show()
    
    def plot_duration_analysis(self, save_path: str = 'visualizations/duration_analysis.png'):
        """Create subplots analyzing duration patterns"""
        if self.df is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Movie duration histogram
        movies = self.df[self.df['type'] == 'Movie'].copy()
        movie_durations = []
        for duration in movies['duration'].dropna():
            if 'min' in str(duration):
                try:
                    minutes = int(str(duration).replace(' min', ''))
                    if 30 <= minutes <= 300:  # Reasonable range
                        movie_durations.append(minutes)
                except:
                    pass
        
        if movie_durations:
            ax1.hist(movie_durations, bins=20, color=self.colors[0], alpha=0.7, edgecolor='black')
            ax1.set_title('Movie Duration Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Duration (minutes)', fontsize=12)
            ax1.set_ylabel('Number of Movies', fontsize=12)
            ax1.grid(alpha=0.3)
            ax1.axvline(np.mean(movie_durations), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(movie_durations):.0f} min')
            ax1.legend()
        
        # TV Show seasons bar chart
        tv_shows = self.df[self.df['type'] == 'TV Show'].copy()
        tv_seasons = []
        for duration in tv_shows['duration'].dropna():
            if 'season' in str(duration).lower():
                try:
                    seasons = int(str(duration).split()[0])
                    if 1 <= seasons <= 20:  # Reasonable range
                        tv_seasons.append(seasons)
                except:
                    pass
        
        if tv_seasons:
            season_counts = Counter(tv_seasons)
            seasons = sorted(season_counts.keys())
            counts = [season_counts[s] for s in seasons]
            
            ax2.bar(seasons, counts, color=self.colors[3], alpha=0.7, edgecolor='black')
            ax2.set_title('TV Show Season Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Number of Seasons', fontsize=12)
            ax2.set_ylabel('Number of TV Shows', fontsize=12)
            ax2.grid(axis='y', alpha=0.3)
            ax2.set_xticks(seasons)
        
        plt.tight_layout()
        self._save_plot(save_path)
        plt.show()
    
    def plot_yearly_content_heatmap(self, save_path: str = 'visualizations/yearly_content_heatmap.png'):
        """Create heatmap showing content addition patterns by year and month"""
        if self.df is None:
            return
        
        # Parse date_added column
        self.df['date_added'] = pd.to_datetime(self.df['date_added'], errors='coerce')
        
        # Filter for valid dates
        valid_dates = self.df[self.df['date_added'].notna()].copy()
        
        if len(valid_dates) == 0:
            print("❌ No valid date_added data found for heatmap")
            return
        
        # Extract year and month
        valid_dates['add_year'] = valid_dates['date_added'].dt.year
        valid_dates['add_month'] = valid_dates['date_added'].dt.month
        
        # Create pivot table
        heatmap_data = valid_dates.groupby(['add_year', 'add_month']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(14, 8))
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Reds', 
                   cbar_kws={'label': 'Number of Titles Added'})
        
        plt.title('Netflix Content Addition Patterns (Year vs Month)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Month', fontsize=12, fontweight='bold')
        plt.ylabel('Year', fontsize=12, fontweight='bold')
        
        # Set month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(range(12), month_labels)
        
        plt.tight_layout()
        self._save_plot(save_path)
        plt.show()
    
    def plot_top_directors(self, top_n: int = 15, save_path: str = 'visualizations/top_directors.png'):
        """Create bar chart of most prolific directors"""
        if self.df is None:
            return
        
        # Process directors (handle multiple directors per title)
        all_directors = []
        for directors_str in self.df['director'].dropna():
            if str(directors_str) not in ['Unknown Director', 'nan']:
                directors = [director.strip() for director in str(directors_str).split(',')]
                all_directors.extend(directors)
        
        director_counts = Counter(all_directors)
        top_directors = dict(director_counts.most_common(top_n))
        
        plt.figure(figsize=(12, 10))
        directors = list(top_directors.keys())
        counts = list(top_directors.values())
        
        bars = plt.barh(range(len(directors)), counts, color=self.colors[1])
        
        plt.yticks(range(len(directors)), directors)
        plt.xlabel('Number of Titles', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Most Prolific Directors on Netflix', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(count + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{count}', va='center', fontweight='bold')
        
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        self._save_plot(save_path)
        plt.show()
    
    def plot_correlation_matrix(self, save_path: str = 'visualizations/correlation_matrix.png'):
        """Create correlation matrix for numerical features"""
        if self.df is None:
            return
        
        # Prepare numerical data
        numerical_df = self.df.copy()
        
        # Convert categorical to numerical for correlation
        numerical_df['is_movie'] = (numerical_df['type'] == 'Movie').astype(int)
        numerical_df['content_age'] = 2025 - numerical_df['release_year']
        
        # Extract duration in minutes for movies
        numerical_df['duration_minutes'] = 0
        for idx, row in numerical_df.iterrows():
            if 'min' in str(row['duration']):
                try:
                    minutes = int(str(row['duration']).replace(' min', ''))
                    numerical_df.loc[idx, 'duration_minutes'] = minutes
                except:
                    pass
        
        # Select features for correlation
        corr_features = ['release_year', 'is_movie', 'content_age', 'duration_minutes']
        corr_data = numerical_df[corr_features].corr()
        
        plt.figure(figsize=(10, 8))
        
        # Create correlation heatmap
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        self._save_plot(save_path)
        plt.show()
    
    def create_dashboard_summary(self, save_path: str = 'visualizations/dashboard_summary.png'):
        """Create a comprehensive dashboard-style summary visualization"""
        if self.df is None:
            return
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Content Type Pie Chart
        ax1 = fig.add_subplot(gs[0, 0])
        type_counts = self.df['type'].value_counts()
        ax1.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
               colors=[self.colors[0], self.colors[3]], startangle=90)
        ax1.set_title('Content Type Distribution', fontweight='bold')
        
        # 2. Top 5 Countries
        ax2 = fig.add_subplot(gs[0, 1])
        all_countries = []
        for countries_str in self.df['country'].dropna():
            if str(countries_str) not in ['Unknown Country', 'nan']:
                countries = [country.strip() for country in str(countries_str).split(',')]
                all_countries.extend(countries)
        country_counts = Counter(all_countries)
        top_5_countries = dict(country_counts.most_common(5))
        
        ax2.bar(range(len(top_5_countries)), list(top_5_countries.values()), 
               color=self.colors[0])
        ax2.set_xticks(range(len(top_5_countries)))
        ax2.set_xticklabels(list(top_5_countries.keys()), rotation=45, ha='right')
        ax2.set_title('Top 5 Countries', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Content Trends
        ax3 = fig.add_subplot(gs[0, 2:])
        yearly_content = self.df.groupby(['release_year', 'type']).size().unstack(fill_value=0)
        ax3.plot(yearly_content.index, yearly_content['Movie'], 
                color=self.colors[0], linewidth=2, label='Movies')
        ax3.plot(yearly_content.index, yearly_content['TV Show'], 
                color=self.colors[3], linewidth=2, label='TV Shows')
        ax3.set_title('Content Release Trends', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        ax3.set_xlim(1990, 2025)
        
        # 4. Top 5 Genres
        ax4 = fig.add_subplot(gs[1, :2])
        all_genres = []
        for genres_str in self.df['listed_in'].dropna():
            if str(genres_str) not in ['Unknown Genre', 'nan']:
                genres = [genre.strip() for genre in str(genres_str).split(',')]
                all_genres.extend(genres)
        genre_counts = Counter(all_genres)
        top_5_genres = dict(genre_counts.most_common(5))
        
        ax4.barh(range(len(top_5_genres)), list(top_5_genres.values()), 
                color=self.colors[1])
        ax4.set_yticks(range(len(top_5_genres)))
        ax4.set_yticklabels(list(top_5_genres.keys()))
        ax4.set_title('Top 5 Genres', fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        
        # 5. Rating Distribution
        ax5 = fig.add_subplot(gs[1, 2:])
        rating_counts = self.df['rating'].value_counts()
        ax5.bar(range(len(rating_counts)), rating_counts.values, color=self.colors[3])
        ax5.set_xticks(range(len(rating_counts)))
        ax5.set_xticklabels(rating_counts.index, rotation=45, ha='right')
        ax5.set_title('Content Rating Distribution', fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Statistics Summary
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Calculate key statistics
        total_titles = len(self.df)
        movies = len(self.df[self.df['type'] == 'Movie'])
        tv_shows = len(self.df[self.df['type'] == 'TV Show'])
        countries = self.df['country'].nunique()
        years_span = f"{self.df['release_year'].min()}-{self.df['release_year'].max()}"
        
        stats_text = f"""
        KEY STATISTICS
        
        Total Titles: {total_titles:,}                Movies: {movies:,} ({movies/total_titles*100:.1f}%)
        TV Shows: {tv_shows:,} ({tv_shows/total_titles*100:.1f}%)            Countries: {countries:,}
        Year Range: {years_span}                        Unique Directors: {self.df['director'].nunique():,}
        """
        
        ax6.text(0.1, 0.5, stats_text, fontsize=14, fontweight='bold',
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor=self.colors[2], alpha=0.8))
        
        # Main title
        fig.suptitle('Netflix Content Analysis Dashboard', fontsize=24, fontweight='bold', y=0.95)
        
        self._save_plot(save_path)
        plt.show()
    
    def _save_plot(self, filepath: str):
        """Save plot to file"""
        try:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"💾 Plot saved: {filepath}")
        except Exception as e:
            print(f"❌ Error saving plot: {e}")
    
    def generate_all_visualizations(self):
        """Generate all visualizations in sequence"""
        print("🎨 Generating Netflix Data Visualizations...")
        print("="*60)
        
        # Load data if not already loaded
        if self.df is None:
            self.load_data()
            if self.df is None:
                return
        
        print("\n📊 Creating visualizations...")
        
        try:
            # Generate all plots
            self.plot_content_type_distribution()
            self.plot_content_trends_over_time()
            self.plot_top_countries()
            self.plot_genre_distribution()
            self.plot_rating_distribution()
            self.plot_duration_analysis()
            self.plot_yearly_content_heatmap()
            self.plot_top_directors()
            self.plot_correlation_matrix()
            self.create_dashboard_summary()
            
            print("\n✅ All visualizations generated successfully!")
            print("\nNext steps:")
            print("1. Check 'visualizations/' folder for all charts")
            print("2. Run 4_dashboard.py for interactive dashboard")
            print("3. Run 5_machine_learning.py for advanced analysis")
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")


def main():
    """Main function to run visualization generation"""
    print("🎬 Netflix Data Visualization Module")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = NetflixVisualizer()
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()
    
    return visualizer


if __name__ == "__main__":
    main()