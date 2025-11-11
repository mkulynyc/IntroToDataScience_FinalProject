import pandas as pd
import numpy as np
from recomender_v1 import NetflixDataProcessor

print('📺 Testing Netflix Data Processing with TV Show Minutes...')
print('=' * 60)

# Create processor and load data
processor = NetflixDataProcessor()
df = processor.load_data('netflix_titles.csv', 'netflix_movies_tv_runtime.csv')

print(f'\n📊 Loaded data shape: {df.shape}')
print(f'Available columns: {list(df.columns)}')

# Clean the data
processor.clean_data()
print(f'\n�� After cleaning: {processor.df.shape}')

# Normalize duration
processor.normalize_duration()
print(f'\n⏱️  After duration normalization: {processor.df.shape}')

# Check TV show data specifically
tv_shows = processor.df[processor.df['type'] == 'TV Show'].copy()
print(f'\n📺 TV Shows found: {len(tv_shows)}')

if len(tv_shows) > 0:
    print('\n🔍 TV Show Duration Analysis:')
    
    # Check if tv_show_minutes column exists and has data
    if 'tv_show_minutes' in tv_shows.columns:
        valid_tv_minutes = tv_shows['tv_show_minutes'].notna().sum()
        print(f'  TV shows with tv_show_minutes data: {valid_tv_minutes}/{len(tv_shows)}')
        if valid_tv_minutes > 0:
            print(f'  Average TV show minutes: {tv_shows["tv_show_minutes"].mean():.1f}')
            print(f'  TV show minutes range: {tv_shows["tv_show_minutes"].min():.0f} - {tv_shows["tv_show_minutes"].max():.0f}')
    
    # Check content_minutes column
    if 'content_minutes' in tv_shows.columns:
        valid_content_minutes = tv_shows['content_minutes'].notna().sum()
        print(f'  TV shows with content_minutes data: {valid_content_minutes}/{len(tv_shows)}')
        if valid_content_minutes > 0:
            print(f'  Average content minutes: {tv_shows["content_minutes"].mean():.1f}')
    
    # Show sample data
    print('\n📋 Sample TV Show Data:')
    sample_cols = ['title', 'tv_show_minutes', 'content_minutes', 'season_count']
    available_cols = [col for col in sample_cols if col in tv_shows.columns]
    if available_cols:
        print(tv_shows[available_cols].head(3).to_string())

else:
    print('❌ No TV shows found after processing!')

print('\n✅ Test completed!')
