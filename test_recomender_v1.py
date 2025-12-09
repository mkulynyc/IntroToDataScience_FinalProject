"""
Unit tests for recomender_v1.py
Tests all major classes and their key methods
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from recomender_v1 import (
    NetflixDataProcessor,
    StatisticalAnalyzer,
    BayesianInference,
    LinearAlgebraAnalyzer,
    RegressionAnalyzer,
    ClusteringAnalyzer,
    TimeSeriesAnalyzer,
    RecommendationEngine,
    NetflixAnalyticsPipeline
)


class TestNetflixDataProcessor(unittest.TestCase):
    """Test cases for NetflixDataProcessor class"""
    
    def setUp(self):
        """Set up test data before each test"""
        self.processor = NetflixDataProcessor()
        # Create sample data
        self.sample_data = pd.DataFrame({
            'show_id': ['s1', 's2', 's3', 's4', 's5'],
            'type': ['Movie', 'TV Show', 'Movie', 'TV Show', 'Movie'],
            'title': ['Movie 1', 'Show 1', 'Movie 2', 'Show 2', 'Movie 3'],
            'director': ['Director A', 'Director B', 'Director A', np.nan, 'Director C'],
            'cast': ['Actor 1, Actor 2', 'Actor 3, Actor 4', 'Actor 1', 'Actor 5', np.nan],
            'country': ['USA', 'UK, France', 'USA', 'Canada', 'India'],
            'date_added': ['January 1, 2020', 'February 15, 2021', 'March 10, 2019', 
                          'April 5, 2022', 'May 20, 2020'],
            'release_year': [2019, 2020, 2018, 2021, 2017],
            'rating': ['PG-13', 'TV-MA', 'R', 'TV-14', 'PG'],
            'duration': ['90 min', '2 Seasons', '120 min', '1 Season', '95 min'],
            'listed_in': ['Action, Drama', 'Comedy, Romance', 'Horror', 'Thriller', 'Drama, Romance'],
            'description': ['An action movie', 'A comedy show', 'Horror film', 'Thriller series', 'Drama movie']
        })
        self.processor.df = self.sample_data.copy()
    
    def test_init(self):
        """Test processor initialization"""
        processor = NetflixDataProcessor()
        self.assertIsNone(processor.df)
        self.assertIsNone(processor.df_listed_in)
        self.assertIsNone(processor.df_country)
        self.assertIsNone(processor.df_cast)
        self.assertIsNone(processor.df_director)
    
    def test_clean_data(self):
        """Test data cleaning functionality"""
        cleaned_df = self.processor.clean_data()
        
        # Check that dataframe is not empty
        self.assertIsNotNone(cleaned_df)
        self.assertGreater(len(cleaned_df), 0)
        
        # Check that date_added is datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned_df['date_added']))
        
        # Check that release_year is numeric
        self.assertTrue(pd.api.types.is_integer_dtype(cleaned_df['release_year']))
        
        # Check that type is standardized
        valid_types = cleaned_df['type'].str.strip().str.lower().isin(['movie', 'tv show'])
        self.assertTrue(valid_types.all())
    
    def test_normalize_duration(self):
        """Test duration normalization"""
        self.processor.clean_data()
        normalized_df = self.processor.normalize_duration()
        
        # Check that duration columns exist
        self.assertIn('movie_minutes', normalized_df.columns)
        self.assertIn('season_count', normalized_df.columns)
        self.assertIn('duration_value', normalized_df.columns)
        
        # Check that movie minutes are extracted for movies
        movies = normalized_df[normalized_df['type'].str.lower() == 'movie']
        if len(movies) > 0:
            self.assertTrue(movies['movie_minutes'].notna().any())
        
        # Check that season counts are extracted for TV shows
        tv_shows = normalized_df[normalized_df['type'].str.lower() == 'tv show']
        if len(tv_shows) > 0:
            self.assertTrue(tv_shows['season_count'].notna().any())
    
    def test_create_exploded_tables(self):
        """Test creation of normalized tables"""
        self.processor.clean_data()
        self.processor.normalize_duration()
        genre_df, country_df, cast_df, director_df = self.processor.create_exploded_tables()
        
        # Check that tables are created
        self.assertIsNotNone(genre_df)
        self.assertIsNotNone(country_df)
        self.assertIsNotNone(cast_df)
        self.assertIsNotNone(director_df)
        
        # Check that genre table has correct columns
        self.assertIn('show_id', genre_df.columns)
        self.assertIn('listed_in', genre_df.columns)
        
        # Check that exploded tables have more rows than original
        self.assertGreaterEqual(len(genre_df), len(self.sample_data))
    
    def test_infer_release_year(self):
        """Test release year inference"""
        df_no_year = self.sample_data.copy()
        df_no_year['release_year'] = np.nan
        self.processor.df = df_no_year
        
        inferred_years = self.processor._infer_release_year()
        
        # Check that years are reasonable
        current_year = datetime.now().year
        self.assertTrue((inferred_years >= 1990).all())
        self.assertTrue((inferred_years <= current_year).all())
    
    def test_ensure_genre_information(self):
        """Test genre information creation"""
        self.processor.clean_data()
        self.processor.normalize_duration()
        
        # Check that genre columns exist
        self.assertIn('genre_list', self.processor.df.columns)
        self.assertIn('primary_genre', self.processor.df.columns)
        
        # Check that genre_list contains lists
        self.assertTrue(all(isinstance(g, list) for g in self.processor.df['genre_list']))


class TestStatisticalAnalyzer(unittest.TestCase):
    """Test cases for StatisticalAnalyzer class"""
    
    def setUp(self):
        """Set up test data"""
        self.df = pd.DataFrame({
            'movie_minutes': [90, 120, 95, 110, 85, 100, 115, 105],
            'release_year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
            'type': ['Movie'] * 8
        })
        self.analyzer = StatisticalAnalyzer(self.df)
    
    def test_init(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer.df)
        self.assertEqual(len(self.analyzer.df), 8)
    
    def test_descriptive_statistics(self):
        """Test descriptive statistics calculation"""
        movie_stats, year_stats = self.analyzer.descriptive_statistics()
        
        # Check that statistics are calculated
        self.assertIsNotNone(movie_stats)
        self.assertIsNotNone(year_stats)
        
        # Check that statistics contain expected values
        self.assertIn('mean', movie_stats.index)
        self.assertIn('std', movie_stats.index)
        self.assertIn('min', movie_stats.index)
        self.assertIn('max', movie_stats.index)


class TestBayesianInference(unittest.TestCase):
    """Test cases for BayesianInference class"""
    
    def setUp(self):
        """Set up test data"""
        self.df = pd.DataFrame({
            'type': ['Movie', 'TV Show', 'Movie', 'TV Show', 'Movie', 'TV Show'],
            'country': ['USA', 'UK', 'USA', 'Canada', 'India', 'UK'],
            'rating': ['PG-13', 'TV-MA', 'R', 'TV-14', 'PG', 'TV-MA'],
            'movie_minutes': [90, 0, 120, 0, 95, 0],
            'season_count': [0, 2, 0, 1, 0, 3]
        })
        self.bayesian = BayesianInference(self.df)
    
    def test_init(self):
        """Test Bayesian inference initialization"""
        self.assertIsNotNone(self.bayesian.df)
        self.assertIsNone(self.bayesian.model)
    
    def test_prepare_bayesian_data(self):
        """Test Bayesian data preparation"""
        X, y = self.bayesian.prepare_bayesian_data()
        
        # Check that data is prepared
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)


class TestLinearAlgebraAnalyzer(unittest.TestCase):
    """Test cases for LinearAlgebraAnalyzer class"""
    
    def setUp(self):
        """Set up test data"""
        self.df = pd.DataFrame({
            'movie_minutes': [90, 120, 95, 110, 85],
            'release_year': [2015, 2016, 2017, 2018, 2019],
            'season_count': [0, 0, 0, 0, 0],
            'listed_in': ['Action', 'Drama', 'Comedy', 'Horror', 'Romance']
        })
        self.analyzer = LinearAlgebraAnalyzer(self.df)
    
    def test_init(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer.df)
        self.assertIsNone(self.analyzer.similarity_matrix)
        self.assertIsNone(self.analyzer.pca_result)


class TestRegressionAnalyzer(unittest.TestCase):
    """Test cases for RegressionAnalyzer class"""
    
    def setUp(self):
        """Set up test data"""
        self.df = pd.DataFrame({
            'release_year': [2015, 2016, 2017, 2018, 2019, 2020],
            'movie_minutes': [90, 95, 100, 105, 110, 115],
            'type': ['Movie'] * 6
        })
        self.analyzer = RegressionAnalyzer(self.df)
    
    def test_init(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer.df)
        self.assertIsNone(self.analyzer.model)


class TestClusteringAnalyzer(unittest.TestCase):
    """Test cases for ClusteringAnalyzer class"""
    
    def setUp(self):
        """Set up test data"""
        self.df = pd.DataFrame({
            'movie_minutes': [90, 95, 120, 125, 85, 88, 130, 135],
            'release_year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
            'type': ['Movie'] * 8
        })
        self.analyzer = ClusteringAnalyzer(self.df)
    
    def test_init(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer.df)
        self.assertIsNone(self.analyzer.kmeans_model)
        self.assertIsNone(self.analyzer.clusters)


class TestTimeSeriesAnalyzer(unittest.TestCase):
    """Test cases for TimeSeriesAnalyzer class"""
    
    def setUp(self):
        """Set up test data"""
        date_range = pd.date_range(start='2015-01-01', periods=24, freq='M')
        self.df = pd.DataFrame({
            'date_added': date_range,
            'show_id': [f's{i}' for i in range(24)],
            'type': ['Movie'] * 12 + ['TV Show'] * 12
        })
        self.analyzer = TimeSeriesAnalyzer(self.df)
    
    def test_init(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer.df)


class TestRecommendationEngine(unittest.TestCase):
    """Test cases for RecommendationEngine class"""
    
    def setUp(self):
        """Set up test data"""
        self.df = pd.DataFrame({
            'show_id': ['s1', 's2', 's3', 's4', 's5'],
            'title': ['Movie 1', 'Movie 2', 'Movie 3', 'Show 1', 'Show 2'],
            'listed_in': ['Action, Drama', 'Action, Comedy', 'Drama, Romance', 
                         'Comedy', 'Horror, Thriller'],
            'description': ['An action movie', 'Action comedy', 'Romantic drama',
                           'Funny show', 'Scary thriller'],
            'director': ['Dir A', 'Dir B', 'Dir A', 'Dir C', 'Dir D'],
            'cast': ['Actor 1, Actor 2', 'Actor 2, Actor 3', 'Actor 1, Actor 4',
                    'Actor 5', 'Actor 6'],
            'country': ['USA', 'USA', 'UK', 'Canada', 'France'],
            'type': ['Movie', 'Movie', 'Movie', 'TV Show', 'TV Show']
        })
        self.engine = RecommendationEngine(self.df)
    
    def test_init(self):
        """Test engine initialization"""
        self.assertIsNotNone(self.engine.df)
        self.assertIsNone(self.engine.similarity_matrix)


class TestNetflixAnalyticsPipeline(unittest.TestCase):
    """Test cases for NetflixAnalyticsPipeline class"""
    
    def setUp(self):
        """Set up test pipeline"""
        self.pipeline = NetflixAnalyticsPipeline()
    
    def test_init(self):
        """Test pipeline initialization"""
        self.assertIsNone(self.pipeline.processor)
        self.assertIsNone(self.pipeline.stats_analyzer)
        self.assertIsNone(self.pipeline.bayesian)
        self.assertIsNone(self.pipeline.linear_algebra)
        self.assertIsNone(self.pipeline.regression)
        self.assertIsNone(self.pipeline.clustering)
        self.assertIsNone(self.pipeline.time_series)
        self.assertIsNone(self.pipeline.recommender)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up sample data file"""
        self.sample_data = pd.DataFrame({
            'show_id': ['s1', 's2', 's3'],
            'type': ['Movie', 'TV Show', 'Movie'],
            'title': ['Test Movie 1', 'Test Show 1', 'Test Movie 2'],
            'director': ['Director A', 'Director B', 'Director A'],
            'cast': ['Actor 1, Actor 2', 'Actor 3', 'Actor 1'],
            'country': ['USA', 'UK', 'USA'],
            'date_added': ['January 1, 2020', 'February 15, 2021', 'March 10, 2019'],
            'release_year': [2019, 2020, 2018],
            'rating': ['PG-13', 'TV-MA', 'R'],
            'duration': ['90 min', '2 Seasons', '120 min'],
            'listed_in': ['Action, Drama', 'Comedy', 'Horror'],
            'description': ['Action movie', 'Comedy show', 'Horror film']
        })
        self.test_file = 'test_netflix_data.csv'
        self.sample_data.to_csv(self.test_file, index=False)
    
    def tearDown(self):
        """Clean up test file"""
        import os
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_full_pipeline(self):
        """Test complete data processing pipeline"""
        processor = NetflixDataProcessor()
        df = processor.load_data(self.test_file)
        
        # Test data loading
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        
        # Test cleaning
        df = processor.clean_data()
        self.assertIsNotNone(df)
        
        # Test normalization
        df = processor.normalize_duration()
        self.assertIn('movie_minutes', df.columns)
        self.assertIn('season_count', df.columns)
        
        # Test exploded tables
        genre_df, country_df, cast_df, director_df = processor.create_exploded_tables()
        self.assertGreater(len(genre_df), 0)
        self.assertGreater(len(country_df), 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        processor = NetflixDataProcessor()
        processor.df = pd.DataFrame()
        
        # Should not raise exception
        result = processor.normalize_duration()
        self.assertIsNotNone(result)
    
    def test_missing_columns(self):
        """Test handling of missing columns"""
        processor = NetflixDataProcessor()
        processor.df = pd.DataFrame({
            'show_id': ['s1'],
            'title': ['Movie 1']
        })
        
        # Should not raise exception
        result = processor.clean_data()
        self.assertIsNotNone(result)
    
    def test_null_values(self):
        """Test handling of null values"""
        processor = NetflixDataProcessor()
        processor.df = pd.DataFrame({
            'show_id': ['s1', 's2', 's3'],
            'type': ['Movie', None, 'TV Show'],
            'title': ['Movie 1', 'Movie 2', None],
            'duration': ['90 min', None, '2 Seasons']
        })
        
        result = processor.clean_data()
        self.assertIsNotNone(result)
    
    def test_invalid_duration_format(self):
        """Test handling of invalid duration formats"""
        processor = NetflixDataProcessor()
        processor.df = pd.DataFrame({
            'show_id': ['s1', 's2'],
            'type': ['Movie', 'TV Show'],
            'duration': ['invalid', '']
        })
        
        processor.clean_data()
        result = processor.normalize_duration()
        self.assertIsNotNone(result)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
