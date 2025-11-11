#!/usr/bin/env python3
"""
Test script to verify that the date_added column error has been fixed.
"""

import pandas as pd
from recomender_v1 import NetflixDataProcessor
import os

def test_data_processing():
    """Test the data processing with potential missing date_added column."""
    print("Testing Netflix Data Processing...")
    print("=" * 50)
    
    # Look for CSV files in the current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ No CSV files found in current directory")
        return False
    
    print(f"📁 Found CSV files: {csv_files}")
    
    try:
        # Initialize the processor
        processor = NetflixDataProcessor()
        print("✅ NetflixDataProcessor initialized successfully")
        
        # Load the first CSV file
        csv_file = csv_files[0]
        print(f"📊 Loading data from {csv_file}...")
        
        # Load data
        data = pd.read_csv(csv_file)
        print(f"✅ Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        print(f"📋 Columns: {list(data.columns)}")
        
        # Check if date_added column exists
        has_date_added = 'date_added' in data.columns
        print(f"📅 Date_added column present: {has_date_added}")
        
        # Process the data
        print("🔄 Processing data...")
        cleaned_data = processor.clean_data(data)
        print(f"✅ Data processed successfully: {cleaned_data.shape[0]} rows after cleaning")
        
        # Test time series analysis
        print("📈 Testing time series analysis...")
        from recomender_v1 import TimeSeriesAnalyzer
        ts_analyzer = TimeSeriesAnalyzer()
        
        trends = ts_analyzer.analyze_content_trends(cleaned_data)
        print(f"✅ Time series analysis completed: {len(trends)} trend points")
        
        print("\n🎉 All tests passed! The date_added error has been fixed.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_data_processing()
    if success:
        print("\n✅ All fixes are working correctly!")
    else:
        print("\n❌ Some issues still remain.")