"""
Netflix Data Analysis Module
============================

This module provides comprehensive exploratory data analysis for the Netflix dataset.
It generates statistical summaries, insights, and prepares data for visualization.

Author: Netflix Analysis Project
Date: October 2025
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class NetflixDataAnalyzer:
    """
    Comprehensive data analyzer for Netflix dataset
    """
    
    def __init__(self, data_path: str = 'data/netflix_cleaned.csv'):
        """
        Initialize the data analyzer
        
        Args:
            data_path (str): Path to the cleaned Netflix CSV file
        """
        self.data_path = data_path
        self.df = None
        self.analysis_results = {}
        
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
    
    def basic_statistics(self) -> dict:
        """Generate basic statistics about the dataset"""
        if self.df is None:
            return {}
        
        stats = {
            'total_titles': len(self.df),
            'movies': len(self.df[self.df['type'] == 'Movie']),
            'tv_shows': len(self.df[self.df['type'] == 'TV Show']),
            'unique_directors': self.df['director'].nunique(),
            'unique_countries': self.df['country'].nunique(),
            'year_range': {
                'earliest': self.df['release_year'].min(),
                'latest': self.df['release_year'].max()
            }
        }
        
        # Calculate percentages
        stats['movie_percentage'] = (stats['movies'] / stats['total_titles']) * 100
        stats['tv_show_percentage'] = (stats['tv_shows'] / stats['total_titles']) * 100
        
        self.analysis_results['basic_stats'] = stats
        return stats
    
    def print_basic_statistics(self):
        """Print basic statistics in a formatted way"""
        stats = self.basic_statistics()
        
        print("\n" + "="*60)
        print("📊 NETFLIX DATASET BASIC STATISTICS")
        print("="*60)
        
        print(f"📺 Total Content: {stats['total_titles']:,} titles")
        print(f"🎬 Movies: {stats['movies']:,} ({stats['movie_percentage']:.1f}%)")
        print(f"📺 TV Shows: {stats['tv_shows']:,} ({stats['tv_show_percentage']:.1f}%)")
        print(f"🎭 Unique Directors: {stats['unique_directors']:,}")
        print(f"🌍 Countries: {stats['unique_countries']:,}")
        print(f"📅 Year Range: {stats['year_range']['earliest']} - {stats['year_range']['latest']}")
        print("="*60)
    
    def analyze_content_trends(self) -> dict:
        """Analyze content trends over time"""
        if self.df is None:
            return {}
        
        # Content by year
        yearly_stats = self.df.groupby('release_year').agg({
            'show_id': 'count',
            'type': lambda x: (x == 'Movie').sum()
        }).rename(columns={'show_id': 'total_content', 'type': 'movies'})
        
        yearly_stats['tv_shows'] = yearly_stats['total_content'] - yearly_stats['movies']
        yearly_stats['movie_ratio'] = yearly_stats['movies'] / yearly_stats['total_content']
        
        # Identify trends
        recent_years = yearly_stats.tail(10)
        trend_analysis = {
            'yearly_content': yearly_stats.to_dict('index'),
            'peak_year': yearly_stats['total_content'].idxmax(),
            'peak_content_count': yearly_stats['total_content'].max(),
            'recent_movie_ratio': recent_years['movie_ratio'].mean(),
            'total_growth': {
                'early_2000s': yearly_stats.loc[2000:2010, 'total_content'].sum() if 2000 in yearly_stats.index else 0,
                '2010s': yearly_stats.loc[2010:2020, 'total_content'].sum() if 2010 in yearly_stats.index else 0,
                '2020s': yearly_stats.loc[2020:, 'total_content'].sum() if 2020 in yearly_stats.index else 0
            }
        }
        
        self.analysis_results['content_trends'] = trend_analysis
        return trend_analysis
    
    def analyze_geographical_distribution(self) -> dict:
        """Analyze content distribution by country"""
        if self.df is None:
            return {}
        
        # Split countries and count
        all_countries = []
        for countries_str in self.df['country'].dropna():
            if str(countries_str) != 'nan':
                countries = [country.strip() for country in str(countries_str).split(',')]
                all_countries.extend(countries)
        
        country_counts = Counter(all_countries)
        
        # Remove unknown/invalid entries
        if 'Unknown Country' in country_counts:
            unknown_count = country_counts.pop('Unknown Country')
        else:
            unknown_count = 0
        
        # Get top countries
        top_countries = dict(country_counts.most_common(20))
        
        geo_analysis = {
            'total_countries': len(country_counts),
            'top_countries': top_countries,
            'unknown_countries': unknown_count,
            'us_content': country_counts.get('United States', 0),
            'international_vs_us': {
                'us_content': country_counts.get('United States', 0),
                'international_content': sum(country_counts.values()) - country_counts.get('United States', 0)
            }
        }
        
        self.analysis_results['geographical'] = geo_analysis
        return geo_analysis
    
    def analyze_genres(self) -> dict:
        """Analyze genre distribution and popularity"""
        if self.df is None:
            return {}
        
        # Split genres and count
        all_genres = []
        for genres_str in self.df['listed_in'].dropna():
            if str(genres_str) != 'nan':
                genres = [genre.strip() for genre in str(genres_str).split(',')]
                all_genres.extend(genres)
        
        genre_counts = Counter(all_genres)
        
        # Remove unknown/invalid entries
        if 'Unknown Genre' in genre_counts:
            unknown_count = genre_counts.pop('Unknown Genre')
        else:
            unknown_count = 0
        
        # Analyze genre trends by type
        movie_genres = []
        tv_genres = []
        
        for _, row in self.df.iterrows():
            if pd.notna(row['listed_in']):
                genres = [genre.strip() for genre in str(row['listed_in']).split(',')]
                if row['type'] == 'Movie':
                    movie_genres.extend(genres)
                else:
                    tv_genres.extend(genres)
        
        genre_analysis = {
            'total_unique_genres': len(genre_counts),
            'top_genres_overall': dict(genre_counts.most_common(15)),
            'top_movie_genres': dict(Counter(movie_genres).most_common(10)),
            'top_tv_genres': dict(Counter(tv_genres).most_common(10)),
            'unknown_genres': unknown_count
        }
        
        self.analysis_results['genres'] = genre_analysis
        return genre_analysis
    
    def analyze_ratings(self) -> dict:
        """Analyze content ratings distribution"""
        if self.df is None:
            return {}
        
        # Overall rating distribution
        rating_counts = self.df['rating'].value_counts()
        
        # Rating by content type
        rating_by_type = self.df.groupby(['type', 'rating']).size().unstack(fill_value=0)
        
        # Age appropriateness categories
        family_friendly = ['G', 'PG', 'TV-G', 'TV-Y', 'TV-Y7', 'TV-PG']
        mature_content = ['R', 'TV-MA', 'NC-17']
        teen_content = ['PG-13', 'TV-14']
        
        rating_analysis = {
            'rating_distribution': rating_counts.to_dict(),
            'rating_by_type': rating_by_type.to_dict(),
            'age_categories': {
                'family_friendly': self.df[self.df['rating'].isin(family_friendly)].shape[0],
                'teen_content': self.df[self.df['rating'].isin(teen_content)].shape[0],
                'mature_content': self.df[self.df['rating'].isin(mature_content)].shape[0]
            }
        }
        
        self.analysis_results['ratings'] = rating_analysis
        return rating_analysis
    
    def analyze_duration_patterns(self) -> dict:
        """Analyze duration patterns for movies and TV shows"""
        if self.df is None:
            return {}
        
        # Separate movies and TV shows
        movies = self.df[self.df['type'] == 'Movie'].copy()
        tv_shows = self.df[self.df['type'] == 'TV Show'].copy()
        
        # Extract numeric duration for movies (in minutes)
        movie_durations = []
        for duration in movies['duration'].dropna():
            if 'min' in str(duration):
                try:
                    minutes = int(str(duration).replace(' min', ''))
                    movie_durations.append(minutes)
                except:
                    pass
        
        # Extract seasons for TV shows
        tv_seasons = []
        for duration in tv_shows['duration'].dropna():
            if 'season' in str(duration).lower():
                try:
                    seasons = int(str(duration).split()[0])
                    tv_seasons.append(seasons)
                except:
                    pass
        
        duration_analysis = {
            'movies': {
                'count': len(movie_durations),
                'avg_duration': np.mean(movie_durations) if movie_durations else 0,
                'min_duration': min(movie_durations) if movie_durations else 0,
                'max_duration': max(movie_durations) if movie_durations else 0,
                'median_duration': np.median(movie_durations) if movie_durations else 0
            },
            'tv_shows': {
                'count': len(tv_seasons),
                'avg_seasons': np.mean(tv_seasons) if tv_seasons else 0,
                'min_seasons': min(tv_seasons) if tv_seasons else 0,
                'max_seasons': max(tv_seasons) if tv_seasons else 0,
                'median_seasons': np.median(tv_seasons) if tv_seasons else 0
            }
        }
        
        self.analysis_results['duration'] = duration_analysis
        return duration_analysis
    
    def analyze_cast_and_crew(self) -> dict:
        """Analyze cast and crew patterns"""
        if self.df is None:
            return {}
        
        # Most prolific directors
        all_directors = []
        for directors_str in self.df['director'].dropna():
            if str(directors_str) not in ['Unknown Director', 'nan']:
                directors = [director.strip() for director in str(directors_str).split(',')]
                all_directors.extend(directors)
        
        director_counts = Counter(all_directors)
        
        # Most frequent cast members
        all_cast = []
        for cast_str in self.df['cast'].dropna():
            if str(cast_str) not in ['Unknown Cast', 'nan']:
                cast_members = [actor.strip() for actor in str(cast_str).split(',')]
                all_cast.extend(cast_members[:5])  # Limit to first 5 to avoid overwhelming data
        
        cast_counts = Counter(all_cast)
        
        crew_analysis = {
            'top_directors': dict(director_counts.most_common(20)),
            'top_cast': dict(cast_counts.most_common(20)),
            'total_unique_directors': len(director_counts),
            'total_unique_cast': len(cast_counts),
            'prolific_directors': {name: count for name, count in director_counts.items() if count >= 5}
        }
        
        self.analysis_results['cast_crew'] = crew_analysis
        return crew_analysis
    
    def generate_insights(self) -> list:
        """Generate key insights from the analysis"""
        if not self.analysis_results:
            print("❌ No analysis results available. Run analysis methods first.")
            return []
        
        insights = []
        
        # Basic insights
        if 'basic_stats' in self.analysis_results:
            stats = self.analysis_results['basic_stats']
            if stats['movie_percentage'] > stats['tv_show_percentage']:
                insights.append(f"Movies dominate Netflix catalog ({stats['movie_percentage']:.1f}% vs {stats['tv_show_percentage']:.1f}% TV shows)")
            else:
                insights.append(f"TV shows are more prevalent ({stats['tv_show_percentage']:.1f}% vs {stats['movie_percentage']:.1f}% movies)")
        
        # Content trends insights
        if 'content_trends' in self.analysis_results:
            trends = self.analysis_results['content_trends']
            insights.append(f"Peak content year was {trends['peak_year']} with {trends['peak_content_count']} titles")
            
            if trends['recent_movie_ratio'] > 0.6:
                insights.append("Recent years show a movie-heavy strategy")
            elif trends['recent_movie_ratio'] < 0.4:
                insights.append("Recent years show increased focus on TV shows")
        
        # Geographical insights
        if 'geographical' in self.analysis_results:
            geo = self.analysis_results['geographical']
            us_ratio = geo['international_vs_us']['us_content'] / (geo['international_vs_us']['us_content'] + geo['international_vs_us']['international_content'])
            if us_ratio > 0.5:
                insights.append(f"US content dominates ({us_ratio*100:.1f}% of total)")
            else:
                insights.append(f"International content is significant ({(1-us_ratio)*100:.1f}% non-US)")
        
        # Genre insights
        if 'genres' in self.analysis_results:
            genres = self.analysis_results['genres']
            top_genre = list(genres['top_genres_overall'].keys())[0]
            insights.append(f"Most popular genre: {top_genre}")
        
        # Rating insights
        if 'ratings' in self.analysis_results:
            ratings = self.analysis_results['ratings']
            age_cats = ratings['age_categories']
            total_content = sum(age_cats.values())
            mature_ratio = age_cats['mature_content'] / total_content
            
            if mature_ratio > 0.3:
                insights.append("High proportion of mature content (30%+ rated R/TV-MA)")
            elif mature_ratio < 0.1:
                insights.append("Family-friendly catalog with minimal mature content")
        
        return insights
    
    def full_analysis(self) -> dict:
        """
        Perform complete data analysis
        
        Returns:
            dict: Complete analysis results
        """
        print("📊 Starting comprehensive data analysis...")
        print("="*60)
        
        # Load data if not already loaded
        if self.df is None:
            self.load_data()
            if self.df is None:
                return {}
        
        print("\n🔍 Running analysis modules...")
        
        # Run all analysis modules
        self.basic_statistics()
        self.analyze_content_trends()
        self.analyze_geographical_distribution()
        self.analyze_genres()
        self.analyze_ratings()
        self.analyze_duration_patterns()
        self.analyze_cast_and_crew()
        
        # Print formatted results
        self.print_basic_statistics()
        self.print_detailed_analysis()
        
        # Generate insights
        insights = self.generate_insights()
        
        print("\n🔍 KEY INSIGHTS:")
        print("-" * 30)
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        print("\n✅ Analysis completed successfully!")
        print("\nNext steps:")
        print("1. Run 3_visualization.py for static charts")
        print("2. Run 4_dashboard.py for interactive dashboard")
        
        return self.analysis_results
    
    def print_detailed_analysis(self):
        """Print detailed analysis results"""
        
        # Content trends
        if 'content_trends' in self.analysis_results:
            trends = self.analysis_results['content_trends']
            print(f"\n📈 CONTENT TRENDS:")
            print(f"Peak year: {trends['peak_year']} ({trends['peak_content_count']} titles)")
            print(f"Recent movie ratio: {trends['recent_movie_ratio']:.1%}")
        
        # Geography
        if 'geographical' in self.analysis_results:
            geo = self.analysis_results['geographical']
            print(f"\n🌍 GEOGRAPHICAL DISTRIBUTION:")
            print(f"Total countries represented: {geo['total_countries']}")
            print("Top 5 countries:")
            for i, (country, count) in enumerate(list(geo['top_countries'].items())[:5], 1):
                print(f"   {i}. {country}: {count:,} titles")
        
        # Genres
        if 'genres' in self.analysis_results:
            genres = self.analysis_results['genres']
            print(f"\n🎭 GENRE ANALYSIS:")
            print(f"Total unique genres: {genres['total_unique_genres']}")
            print("Top 5 genres:")
            for i, (genre, count) in enumerate(list(genres['top_genres_overall'].items())[:5], 1):
                print(f"   {i}. {genre}: {count:,} titles")
        
        # Ratings
        if 'ratings' in self.analysis_results:
            ratings = self.analysis_results['ratings']
            print(f"\n⭐ CONTENT RATINGS:")
            age_cats = ratings['age_categories']
            total = sum(age_cats.values())
            print(f"Family-friendly: {age_cats['family_friendly']:,} ({age_cats['family_friendly']/total:.1%})")
            print(f"Teen content: {age_cats['teen_content']:,} ({age_cats['teen_content']/total:.1%})")
            print(f"Mature content: {age_cats['mature_content']:,} ({age_cats['mature_content']/total:.1%})")
    
    def save_analysis_results(self, filepath: str = 'data/analysis_results.json'):
        """Save analysis results to JSON file"""
        try:
            import json
            import os
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = {}
            for key, value in self.analysis_results.items():
                serializable_results[key] = self._make_serializable(value)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"💾 Analysis results saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving analysis results: {e}")
    
    def _make_serializable(self, obj):
        """Convert numpy types to native Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


def main():
    """Main function to run data analysis"""
    print("🎬 Netflix Data Analysis Module")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = NetflixDataAnalyzer()
    
    # Perform full analysis
    results = analyzer.full_analysis()
    
    if results:
        # Save results
        analyzer.save_analysis_results()
        print("\n🎉 Data analysis completed!")
    
    return analyzer


if __name__ == "__main__":
    main()