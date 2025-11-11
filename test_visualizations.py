#!/usr/bin/env python3
"""
Test script to verify that Netflix data visualizations work correctly
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from recomender_v1 import NetflixDataProcessor

def test_visualizations():
    print("🎨 Testing Netflix Data Visualizations...")
    print("=" * 50)
    
    # Load and process data
    processor = NetflixDataProcessor()
    df = processor.load_data('netflix_titles.csv', 'netflix_movies_tv_runtime.csv')
    processor.clean_data()
    processor.normalize_duration()
    
    print(f"📊 Processed data shape: {processor.df.shape}")
    print(f"Available columns: {list(processor.df.columns)}")
    
    # Test 1: Content type distribution
    print("\n1️⃣ Testing Content Type Distribution...")
    if 'type' in processor.df.columns:
        type_counts = processor.df['type'].value_counts()
        print(f"   ✅ Type counts: {dict(type_counts)}")
        
        try:
            fig_pie = px.pie(
                values=type_counts.values, 
                names=type_counts.index,
                title="Movies vs TV Shows Distribution"
            )
            print("   ✅ Pie chart created successfully")
        except Exception as e:
            print(f"   ❌ Pie chart failed: {e}")
    else:
        print("   ❌ No 'type' column found")
    
    # Test 2: Duration distribution for TV shows
    print("\n2️⃣ Testing TV Show Duration Distribution...")
    if 'type' in processor.df.columns and 'tv_show_minutes' in processor.df.columns:
        tv_shows = processor.df[processor.df['type'] == 'TV Show']
        tv_with_duration = tv_shows.dropna(subset=['tv_show_minutes'])
        
        print(f"   TV shows with duration data: {len(tv_with_duration)}/{len(tv_shows)}")
        
        if len(tv_with_duration) > 0:
            try:
                fig_hist = px.histogram(
                    tv_with_duration, 
                    x='tv_show_minutes',
                    title="TV Show Duration Distribution",
                    nbins=30
                )
                print("   ✅ Duration histogram created successfully")
                print(f"   📊 Duration stats: min={tv_with_duration['tv_show_minutes'].min():.0f}, max={tv_with_duration['tv_show_minutes'].max():.0f}, mean={tv_with_duration['tv_show_minutes'].mean():.0f}")
            except Exception as e:
                print(f"   ❌ Duration histogram failed: {e}")
        else:
            print("   ❌ No TV shows with duration data")
    
    # Test 3: Content over time (if date data available)
    print("\n3️⃣ Testing Content Timeline...")
    if 'date_added' in processor.df.columns:
        date_data = processor.df.dropna(subset=['date_added'])
        print(f"   Records with dates: {len(date_data)}/{len(processor.df)}")
        
        if len(date_data) > 0:
            try:
                # Group by year
                date_data['year_added'] = pd.to_datetime(date_data['date_added']).dt.year
                yearly_counts = date_data.groupby('year_added').size()
                
                fig_trend = px.line(
                    x=yearly_counts.index,
                    y=yearly_counts.values,
                    title="Content Added Over Time"
                )
                print("   ✅ Timeline chart created successfully")
                print(f"   📅 Year range: {yearly_counts.index.min()} - {yearly_counts.index.max()}")
            except Exception as e:
                print(f"   ❌ Timeline chart failed: {e}")
        else:
            print("   ❌ No date data available")
    else:
        print("   ⚠️  No 'date_added' column found")
    
    # Test 4: Sample data display
    print("\n4️⃣ Sample Data for Verification...")
    if 'type' in processor.df.columns:
        print("\n   📺 TV Show Samples:")
        tv_sample = processor.df[processor.df['type'] == 'TV Show'].head(3)
        display_cols = ['title', 'type', 'tv_show_minutes', 'content_minutes', 'season_count']
        available_cols = [col for col in display_cols if col in tv_sample.columns]
        print(tv_sample[available_cols].to_string(index=False))
        
        print("\n   🎬 Movie Samples:")
        movie_sample = processor.df[processor.df['type'] == 'Movie'].head(3)
        display_cols = ['title', 'type', 'movie_minutes', 'content_minutes']
        available_cols = [col for col in display_cols if col in movie_sample.columns]
        print(movie_sample[available_cols].to_string(index=False))
    
    print("\n✅ Visualization test completed!")
    return True

if __name__ == "__main__":
    test_visualizations()