#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared Utility Functions for Netflix Analysis
==============================================

Common functions used across multiple Netflix analysis modules.
This eliminates code duplication and provides a single source of truth.

Author: Netflix Data Analysis Team
Last modified: November 2024
"""

import pandas as pd
import warnings
from pathlib import Path

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')

# Try to import optional dependencies with fallbacks
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("TextBlob not available - emotion analysis will use fallback")

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False
    print("textstat not available - readability analysis will use fallback")

# Constants used across modules
NETFLIX_COLORS = ['#E50914', '#221F1F', '#F5F5F1']
DEFAULT_PORTS = {
    'basic': 8502,
    'enhanced': 8503, 
    'robust': 8501
}

def analyze_emotion(text):
    """
    Analyze emotion/sentiment of text using TextBlob.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Dictionary with polarity, subjectivity, sentiment keys
    """
    if not text or pd.isna(text):
        return {'polarity': 0.0, 'subjectivity': 0.0, 'sentiment': 'neutral'}
    
    if not HAS_TEXTBLOB:
        # Simple fallback sentiment analysis
        text_lower = str(text).lower()
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'best', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return {'polarity': 0.5, 'subjectivity': 0.6, 'sentiment': 'positive'}
        elif neg_count > pos_count:
            return {'polarity': -0.5, 'subjectivity': 0.6, 'sentiment': 'negative'}
        else:
            return {'polarity': 0.0, 'subjectivity': 0.5, 'sentiment': 'neutral'}
    
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Convert polarity to simple categories
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {'polarity': polarity, 'subjectivity': subjectivity, 'sentiment': sentiment}
        
    except Exception as e:
        return {'polarity': 0.0, 'subjectivity': 0.0, 'sentiment': 'neutral'}

def analyze_readability(text):
    """
    Analyze readability of text using Flesch Reading Ease.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Dictionary with score and category keys
    """
    if not text or pd.isna(text):
        return {'score': 50.0, 'category': 'standard'}
    
    if not HAS_TEXTSTAT:
        # Simple fallback readability estimate based on text length and complexity
        text_str = str(text)
        words = len(text_str.split())
        sentences = text_str.count('.') + text_str.count('!') + text_str.count('?') + 1
        avg_words_per_sentence = words / sentences if sentences > 0 else words
        
        # Rough estimation based on sentence length
        if avg_words_per_sentence <= 8:
            return {'score': 80.0, 'category': 'easy'}
        elif avg_words_per_sentence <= 15:
            return {'score': 60.0, 'category': 'standard'}
        else:
            return {'score': 40.0, 'category': 'difficult'}
    
    try:
        clean_text = str(text).strip()
        score = textstat.flesch_reading_ease(clean_text)
        
        # Convert to readable categories
        if score >= 90:
            category = 'very_easy'  # 5th grade
        elif score >= 80:
            category = 'easy'  # 6th grade
        elif score >= 70:
            category = 'fairly_easy'  # 7th grade
        elif score >= 60:
            category = 'standard'  # 8th-9th grade
        elif score >= 50:
            category = 'fairly_difficult'  # 10th-12th grade
        elif score >= 30:
            category = 'difficult'  # college level
        else:
            category = 'very_difficult'  # graduate level
            
        return {'score': score, 'category': category}
        
    except Exception as e:
        return {'score': 50.0, 'category': 'standard'}

def count_words(text):
    """Simple word counter for descriptions."""
    if not text or pd.isna(text):
        return 0
    return len(str(text).split())

def load_netflix_data(verbose=True):
    """
    Load Netflix data from various possible locations.
    
    Args:
        verbose (bool): Print status messages
        
    Returns:
        pandas.DataFrame or None: Netflix data if found
    """
    # Try different possible file locations
    preferred_files = [
        "netflix_with_enhanced_description_analysis.csv",
        "netflix_with_description_analysis.csv", 
        "netflix_enriched_with_tmdb.csv",
        "netflix_clean.csv"
    ]
    
    # Try different possible directories
    search_paths = [
        Path(__file__).parent.parent / "data",
        Path("data"),
        Path("../data"),
        Path("../../data"),
        Path.cwd() / "data"
    ]
    
    for data_dir in search_paths:
        if data_dir.exists():
            for filename in preferred_files:
                file_path = data_dir / filename
                if file_path.exists():
                    return load_single_file(file_path, verbose)
    
    # If no preferred files found, try any CSV in data directories
    for data_dir in search_paths:
        if data_dir.exists():
            csv_files = list(data_dir.glob("netflix*.csv"))
            if csv_files:
                return load_single_file(csv_files[0], verbose)
    
    if verbose:
        print("No Netflix data files found in any expected location")
    return None

def load_single_file(data_file, verbose=True):
    """Load a single Netflix data file with error handling."""
    if data_file and Path(data_file).exists():
        try:
            df = pd.read_csv(data_file)
            if verbose:
                print(f"Loaded {len(df):,} records from {data_file.name}")
            return df
        except Exception as e:
            if verbose:
                print(f"Error loading {data_file}: {e}")
            return None
    else:
        if verbose:
            print("No Netflix data files found")
        return None

def get_streamlit_color_classes():
    """Return CSS classes for consistent Streamlit styling."""
    return """
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #E50914;
            text-align: center;
            margin-bottom: 2rem;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 1rem;
        }
        .emotion-positive {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
        }
        .emotion-negative {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
        }
        .section-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #E50914;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .metric-box {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
        .original-badge {
            background-color: #e8f5e8;
            border-left: 4px solid #28a745;
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 4px;
        }
        .licensed-badge {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 4px;
        }
    """

def validate_netflix_dataframe(df):
    """
    Validate that a DataFrame contains the expected Netflix data structure.
    
    Args:
        df (pandas.DataFrame): DataFrame to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty or None"
    
    # Check for required columns
    required_columns = ['title', 'description']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    return True, "DataFrame is valid"

def safe_port_assignment(base_port=8501):
    """
    Get an available port for Streamlit, starting from base_port.
    
    Args:
        base_port (int): Starting port number
        
    Returns:
        int: Available port number
    """
    import socket
    
    for port in range(base_port, base_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    return 0