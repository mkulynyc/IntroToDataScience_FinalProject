"""
Netflix Data Cleaning Module
============================

This module provides comprehensive data cleaning functionality for the Netflix dataset.
It handles missing values, unknown entries, duplicates, and data validation.

Author: Netflix Analysis Project
Date: October 2025
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class NetflixDataCleaner:
    """
    A comprehensive data cleaner for Netflix dataset
    """
    
    def __init__(self, data_path: str = 'netflix_titles.csv'):
        """
        Initialize the data cleaner
        
        Args:
            data_path (str): Path to the Netflix CSV file
        """
        self.data_path = data_path
        self.df = None
        self.original_shape = None
        self.cleaning_log = []
        
    def load_data(self) -> pd.DataFrame:
        """Load the Netflix dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            self.original_shape = self.df.shape
            self.log_action(f"Loaded data with shape {self.original_shape}")
            print(f"✓ Data loaded successfully: {self.original_shape[0]} rows, {self.original_shape[1]} columns")
            return self.df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def log_action(self, action: str):
        """Log cleaning actions"""
        self.cleaning_log.append(action)
    
    def analyze_data_quality(self) -> Dict:
        """
        Analyze data quality issues
        
        Returns:
            Dict: Summary of data quality issues
        """
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return {}
        
        quality_report = {}
        
        # Missing values
        missing_values = self.df.isnull().sum()
        quality_report['missing_values'] = missing_values[missing_values > 0].to_dict()
        
        # Unknown values
        unknown_counts = {}
        for col in self.df.select_dtypes(include=['object']).columns:
            unknown_count = self.df[col].astype(str).str.lower().str.contains('unknown', na=False).sum()
            if unknown_count > 0:
                unknown_counts[col] = unknown_count
        quality_report['unknown_values'] = unknown_counts
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        quality_report['duplicates'] = duplicates
        
        # Data types
        quality_report['data_types'] = self.df.dtypes.to_dict()
        
        return quality_report
    
    def print_quality_report(self):
        """Print a comprehensive data quality report"""
        report = self.analyze_data_quality()
        
        print("\n" + "="*60)
        print("📊 NETFLIX DATA QUALITY REPORT")
        print("="*60)
        
        print(f"📈 Dataset Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        
        # Missing values
        if report['missing_values']:
            print("\n🔍 Missing Values:")
            for col, count in report['missing_values'].items():
                pct = (count / len(self.df)) * 100
                print(f"   • {col}: {count:,} ({pct:.1f}%)")
        else:
            print("\n✓ No missing values found")
        
        # Unknown values
        if report['unknown_values']:
            print("\n❓ Unknown Values:")
            for col, count in report['unknown_values'].items():
                pct = (count / len(self.df)) * 100
                print(f"   • {col}: {count:,} ({pct:.1f}%)")
        else:
            print("\n✓ No unknown values found")
        
        # Duplicates
        if report['duplicates'] > 0:
            pct = (report['duplicates'] / len(self.df)) * 100
            print(f"\n🔄 Duplicates: {report['duplicates']:,} ({pct:.1f}%)")
        else:
            print("\n✓ No duplicates found")
        
        print("="*60)
    
    def clean_missing_values(self):
        """Handle missing values in the dataset"""
        if self.df is None:
            print("❌ No data loaded")
            return
        
        initial_nulls = self.df.isnull().sum().sum()
        
        # Fill missing values with appropriate defaults
        fill_values = {
            'director': 'Unknown Director',
            'cast': 'Unknown Cast',
            'country': 'Unknown Country',
            'date_added': 'Unknown Date',
            'rating': 'Not Rated',
            'duration': 'Unknown Duration',
            'listed_in': 'Unknown Genre',
            'description': 'No description available'
        }
        
        for column, fill_value in fill_values.items():
            if column in self.df.columns:
                before = self.df[column].isnull().sum()
                self.df[column] = self.df[column].fillna(fill_value)
                if before > 0:
                    self.log_action(f"Filled {before} missing values in {column}")
        
        final_nulls = self.df.isnull().sum().sum()
        print(f"✓ Handled missing values: {initial_nulls} → {final_nulls}")
    
    def fix_unknown_values(self):
        """Fix and improve 'Unknown' values where possible"""
        if self.df is None:
            print("❌ No data loaded")
            return
        
        # Country mapping based on patterns and common knowledge
        country_mappings = {
            'united states': 'United States',
            'uk': 'United Kingdom',
            'united kingdom': 'United Kingdom',
            'india': 'India',
            'canada': 'Canada',
            'australia': 'Australia',
            'france': 'France',
            'germany': 'Germany',
            'japan': 'Japan',
            'south korea': 'South Korea',
            'brazil': 'Brazil',
            'mexico': 'Mexico',
            'spain': 'Spain',
            'italy': 'Italy',
            'netflix': 'United States'  # Netflix originals default to US
        }
        
        # Fix country values
        if 'country' in self.df.columns:
            for old, new in country_mappings.items():
                mask = self.df['country'].astype(str).str.lower().str.contains(old, na=False)
                if mask.any():
                    self.df.loc[mask, 'country'] = new
        
        # Fix ratings
        if 'rating' in self.df.columns:
            # Map common rating patterns
            rating_mappings = {
                'tv-ma': 'TV-MA',
                'tv-14': 'TV-14',
                'tv-pg': 'TV-PG',
                'tv-g': 'TV-G',
                'tv-y': 'TV-Y',
                'tv-y7': 'TV-Y7',
                'r': 'R',
                'pg-13': 'PG-13',
                'pg': 'PG',
                'g': 'G',
                'nc-17': 'NC-17'
            }
            
            for old, new in rating_mappings.items():
                mask = self.df['rating'].astype(str).str.lower() == old
                if mask.any():
                    self.df.loc[mask, 'rating'] = new
        
        print("✓ Fixed unknown values with pattern matching")
    
    def clean_duration_format(self):
        """Standardize duration format"""
        if 'duration' in self.df.columns:
            # Standardize duration formats
            def standardize_duration(duration):
                if pd.isna(duration):
                    return 'Unknown Duration'
                
                duration_str = str(duration).strip()
                
                # Extract numbers and units
                if 'min' in duration_str.lower():
                    return duration_str
                elif 'season' in duration_str.lower():
                    return duration_str
                elif duration_str.isdigit():
                    return f"{duration_str} min"
                else:
                    return duration_str
            
            self.df['duration'] = self.df['duration'].apply(standardize_duration)
            print("✓ Standardized duration format")
    
    def remove_duplicates(self):
        """Remove duplicate entries"""
        if self.df is None:
            print("❌ No data loaded")
            return
        
        initial_count = len(self.df)
        
        # Remove exact duplicates
        self.df = self.df.drop_duplicates()
        
        # Remove duplicates based on title and type (more conservative)
        self.df = self.df.drop_duplicates(subset=['title', 'type'], keep='first')
        
        final_count = len(self.df)
        removed = initial_count - final_count
        
        if removed > 0:
            self.log_action(f"Removed {removed} duplicate entries")
            print(f"✓ Removed {removed} duplicates")
        else:
            print("✓ No duplicates found")
    
    def clean_text_fields(self):
        """Clean and standardize text fields"""
        if self.df is None:
            print("❌ No data loaded")
            return
        
        text_columns = ['title', 'director', 'cast', 'country', 'listed_in', 'description']
        
        for col in text_columns:
            if col in self.df.columns:
                # Strip whitespace
                self.df[col] = self.df[col].astype(str).str.strip()
                
                # Remove extra spaces
                self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)
                
                # Fix common encoding issues
                self.df[col] = self.df[col].str.replace('â€™', "'", regex=False)
                self.df[col] = self.df[col].str.replace('â€œ', '"', regex=False)
                self.df[col] = self.df[col].str.replace('â€\x9d', '"', regex=False)
        
        print("✓ Cleaned text fields")
    
    def validate_data(self) -> bool:
        """
        Validate the cleaned data
        
        Returns:
            bool: True if data passes validation
        """
        if self.df is None:
            print("❌ No data loaded")
            return False
        
        validation_passed = True
        print("\n🔍 Validating cleaned data...")
        
        # Check for required columns
        required_columns = ['show_id', 'type', 'title', 'director', 'cast', 'country', 
                           'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description']
        
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            validation_passed = False
        else:
            print("✓ All required columns present")
        
        # Check data types
        if 'release_year' in self.df.columns:
            if not pd.api.types.is_numeric_dtype(self.df['release_year']):
                print("❌ release_year should be numeric")
                validation_passed = False
            else:
                print("✓ release_year is numeric")
        
        # Check for reasonable year ranges
        if 'release_year' in self.df.columns:
            min_year = self.df['release_year'].min()
            max_year = self.df['release_year'].max()
            current_year = 2025
            
            if min_year < 1900 or max_year > current_year:
                print(f"⚠️  Unusual year range: {min_year} - {max_year}")
            else:
                print(f"✓ Reasonable year range: {min_year} - {max_year}")
        
        # Check for empty strings after cleaning
        empty_counts = {}
        for col in self.df.select_dtypes(include=['object']).columns:
            empty_count = (self.df[col] == '').sum()
            if empty_count > 0:
                empty_counts[col] = empty_count
        
        if empty_counts:
            print(f"⚠️  Empty strings found: {empty_counts}")
        else:
            print("✓ No empty strings found")
        
        return validation_passed
    
    def full_clean(self, save_path: str = 'data/netflix_cleaned.csv') -> pd.DataFrame:
        """
        Perform complete data cleaning pipeline
        
        Args:
            save_path (str): Path to save cleaned data
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        print("🧹 Starting comprehensive data cleaning...")
        print("="*50)
        
        # Load data if not already loaded
        if self.df is None:
            self.load_data()
        
        # Print initial quality report
        self.print_quality_report()
        
        # Perform cleaning steps
        print("\n🔧 Cleaning Steps:")
        print("-" * 30)
        
        self.clean_missing_values()
        self.fix_unknown_values()
        self.clean_duration_format()
        self.clean_text_fields()
        self.remove_duplicates()
        
        # Validate cleaned data
        validation_passed = self.validate_data()
        
        # Print final summary
        print("\n📋 Cleaning Summary:")
        print("-" * 30)
        print(f"Original shape: {self.original_shape}")
        print(f"Final shape: {self.df.shape}")
        rows_removed = self.original_shape[0] - self.df.shape[0]
        print(f"Rows removed: {rows_removed}")
        print(f"Data reduction: {(rows_removed/self.original_shape[0]*100):.2f}%")
        
        if validation_passed:
            print("✅ Data cleaning completed successfully!")
        else:
            print("⚠️  Data cleaning completed with warnings")
        
        # Save cleaned data
        try:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.df.to_csv(save_path, index=False)
            print(f"💾 Cleaned data saved to: {save_path}")
        except Exception as e:
            print(f"❌ Error saving data: {e}")
        
        return self.df
    
    def get_cleaning_log(self) -> List[str]:
        """Get the cleaning log"""
        return self.cleaning_log


def main():
    """Main function to run data cleaning"""
    print("🎬 Netflix Data Cleaning Module")
    print("=" * 50)
    
    # Initialize cleaner
    cleaner = NetflixDataCleaner('netflix_titles.csv')
    
    # Perform full cleaning
    cleaned_data = cleaner.full_clean()
    
    if cleaned_data is not None:
        print("\n🎉 Data cleaning completed!")
        print("\nNext steps:")
        print("1. Run 2_data_analysis.py for exploratory analysis")
        print("2. Run 3_visualization.py for static visualizations")
        print("3. Run 4_dashboard.py for interactive dashboard")
    
    return cleaned_data


if __name__ == "__main__":
    main()