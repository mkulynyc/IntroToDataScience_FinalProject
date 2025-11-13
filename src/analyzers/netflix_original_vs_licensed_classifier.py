"""
Netflix Original vs Licensed Content Classifier
==============================================

Machine learning classifier to identify Netflix Originals vs Licensed content
Uses ensemble of rule-based pattern matching approaches for classification.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import streamlit as st

class NetflixContentClassifier:
    def __init__(self):
        """Initialize the Netflix content classifier"""
        
        # Known Netflix Original indicators
        self.original_indicators = {
            'title_patterns': [
                r'netflix',
                r'stranger things',
                r'house of cards',
                r'orange is the new black',
                r'narcos',
                r'the crown',
                r'black mirror',
                r'ozark',
                r'mindhunter',
                r'bojack horseman',
                r'big mouth',
                r'elite',
                r'money heist',
                r'dark',
                r'altered carbon',
                r'russian doll',
                r'the umbrella academy'
            ],
            'description_patterns': [
                r'netflix original',
                r'netflix series',
                r'netflix film',
                r'netflix production',
                r'netflix exclusive'
            ],
            'production_countries': [
                'netflix global',
                'netflix international'
            ]
        }
        
        # Licensed content indicators
        self.licensed_indicators = {
            'title_patterns': [
                r'friends',
                r'the office',
                r'breaking bad',
                r'better call saul',
                r'gray\'s anatomy',
                r'criminal minds',
                r'ncis',
                r'supernatural',
                r'the walking dead'
            ],
            'description_patterns': [
                r'originally aired',
                r'broadcast on',
                r'premiered on',
                r'syndicated',
                r'licensed from'
            ]
        }
        
        # Weights for ensemble classification
        self.weights = {
            'title': 0.25,
            'description': 0.30,
            'year': 0.20,
            'country': 0.15,
            'genre': 0.10
        }

    def extract_features(self, df):
        """Extract features needed for classification"""
        features_df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['title', 'description', 'release_year', 'country', 'listed_in']
        for col in required_cols:
            if col not in features_df.columns:
                features_df[col] = ''
        
        # Fill NaN values
        features_df = features_df.fillna('')
        
        return features_df

    def classify_by_title_patterns(self, df):
        """Classify based on title patterns"""
        scores = []
        
        for _, row in df.iterrows():
            title = str(row.get('title', '')).lower()
            score = 0.0
            
            # Check for Netflix Original patterns
            for pattern in self.original_indicators['title_patterns']:
                if re.search(pattern, title):
                    score += 0.3
            
            # Check for licensed content patterns
            for pattern in self.licensed_indicators['title_patterns']:
                if re.search(pattern, title):
                    score -= 0.4
            
            # Normalize score
            score = max(-1.0, min(1.0, score))
            scores.append(score)
        
        return np.array(scores)

    def classify_by_description(self, df):
        """Classify based on description patterns"""
        scores = []
        
        for _, row in df.iterrows():
            description = str(row.get('description', '')).lower()
            score = 0.0
            
            # Check for Netflix Original patterns
            for pattern in self.original_indicators['description_patterns']:
                if re.search(pattern, description):
                    score += 0.5
            
            # Check for licensed content patterns  
            for pattern in self.licensed_indicators['description_patterns']:
                if re.search(pattern, description):
                    score -= 0.6
            
            # Normalize score
            score = max(-1.0, min(1.0, score))
            scores.append(score)
        
        return np.array(scores)

    def classify_by_release_year(self, df):
        """Classify based on release year patterns"""
        scores = []
        
        for _, row in df.iterrows():
            year = row.get('release_year', 0)
            score = 0.0
            
            try:
                year = int(year)
                
                # Netflix started heavy original content production around 2012-2013
                if year >= 2015:
                    score += 0.3  # More likely to be original
                elif year >= 2012:
                    score += 0.1  # Slightly more likely
                elif year <= 2010:
                    score -= 0.2  # More likely licensed older content
                
            except (ValueError, TypeError):
                score = 0.0  # Unknown year
            
            # Normalize score
            score = max(-1.0, min(1.0, score))
            scores.append(score)
        
        return np.array(scores)

    def classify_by_country_patterns(self, df):
        """Classify based on production country"""
        scores = []
        
        for _, row in df.iterrows():
            country = str(row.get('country', '')).lower()
            score = 0.0
            
            # Netflix tends to produce more originals in certain regions
            netflix_heavy_countries = [
                'united states', 'south korea', 'spain', 'india', 'japan', 
                'canada', 'france', 'germany', 'brazil', 'mexico'
            ]
            
            # Check if country indicates Netflix production
            for netflix_country in netflix_heavy_countries:
                if netflix_country in country:
                    score += 0.2
                    break
            
            # International co-productions often Netflix originals
            if ',' in country and len(country.split(',')) > 2:
                score += 0.1
            
            # Normalize score
            score = max(-1.0, min(1.0, score))
            scores.append(score)
        
        return np.array(scores)

    def classify_by_genre_patterns(self, df):
        """Classify based on genre patterns"""
        scores = []
        
        for _, row in df.iterrows():
            genres = str(row.get('listed_in', '')).lower()
            score = 0.0
            
            # Netflix original genre indicators
            netflix_genres = [
                'limited series', 'netflix original', 'stand-up comedy',
                'reality tv', 'talk show', 'variety show'
            ]
            
            # Licensed content genre indicators
            licensed_genres = [
                'classic', 'sitcom', 'soap opera', 'game show'
            ]
            
            # Check genres
            for genre in netflix_genres:
                if genre in genres:
                    score += 0.2
            
            for genre in licensed_genres:
                if genre in genres:
                    score -= 0.2
            
            # Normalize score
            score = max(-1.0, min(1.0, score))
            scores.append(score)
        
        return np.array(scores)

    def ensemble_classification(self, df):
        """Run ensemble classification using all methods"""
        
        # Extract features
        features_df = self.extract_features(df)
        
        # Run individual classifiers
        title_scores = self.classify_by_title_patterns(features_df)
        desc_scores = self.classify_by_description(features_df)
        year_scores = self.classify_by_release_year(features_df)
        country_scores = self.classify_by_country_patterns(features_df)
        genre_scores = self.classify_by_genre_patterns(features_df)
        
        # Weighted ensemble
        final_scores = (
            title_scores * self.weights['title'] +
            desc_scores * self.weights['description'] +
            year_scores * self.weights['year'] +
            country_scores * self.weights['country'] +
            genre_scores * self.weights['genre']
        )
        
        # Convert scores to classifications
        classifications = []
        confidences = []
        
        for score in final_scores:
            if score > 0.1:
                classification = 'Netflix Original'
            elif score < -0.1:
                classification = 'Licensed Content'
            else:
                classification = 'Uncertain'
            
            confidence = min(abs(score) * 2, 1.0)
            
            classifications.append(classification)
            confidences.append(confidence)
        
        # Add results to dataframe
        result_df = df.copy()
        result_df['content_classification'] = classifications
        result_df['classification_confidence'] = confidences
        result_df['classification_score'] = final_scores
        
        # Add component scores for analysis
        result_df['title_score'] = title_scores
        result_df['description_score'] = desc_scores
        result_df['year_score'] = year_scores
        result_df['country_score'] = country_scores
        result_df['genre_score'] = genre_scores
        
        return result_df

def main():
    """Test the classifier with sample data"""
    print("Netflix Original vs Licensed Content Classifier")
    print("=" * 50)
    
    # Sample test data
    test_data = pd.DataFrame({
        'title': ['Stranger Things', 'Friends', 'The Crown', 'Breaking Bad', 'Dark'],
        'description': ['Netflix Original series', 'Classic sitcom', 'Netflix historical drama', 'Crime drama series', 'Netflix German series'],
        'release_year': [2016, 1994, 2016, 2008, 2017],
        'country': ['United States', 'United States', 'United Kingdom', 'United States', 'Germany'],
        'listed_in': ['TV Thrillers, TV Sci-Fi', 'Sitcom, Classic TV', 'Historical Drama, Netflix Original', 'Crime Drama', 'Sci-Fi, Netflix Original']
    })
    
    # Initialize classifier
    classifier = NetflixContentClassifier()
    
    # Run classification
    results = classifier.ensemble_classification(test_data)
    
    # Display results
    print("\nClassification Results:")
    for _, row in results.iterrows():
        print(f"Title: {row['title']}")
        print(f"Classification: {row['content_classification']}")
        print(f"Confidence: {row['classification_confidence']:.2f}")
        print(f"Score: {row['classification_score']:.3f}")
        print("-" * 30)

if __name__ == "__main__":
    main()
