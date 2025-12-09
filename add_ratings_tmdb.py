"""
Script to add TMDB ratings to netflix_movies_tv_runtime.csv
Fetches vote_average (rating) from TMDB API for each movie and TV show
"""

import pandas as pd
import requests
import time
from typing import Optional

# TMDB API Configuration
TMDB_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your TMDB API key
TMDB_BASE_URL = "https://api.themoviedb.org/3"

def get_tmdb_rating(tmdb_id: int, content_type: str) -> Optional[float]:
    """
    Fetch rating (vote_average) from TMDB API
    
    Args:
        tmdb_id: TMDB ID of the content
        content_type: 'Movie' or 'TV Show'
    
    Returns:
        Rating (vote_average) as float, or None if not found
    """
    if pd.isna(tmdb_id):
        return None
    
    # Convert content_type to TMDB endpoint type
    endpoint_type = "movie" if content_type == "Movie" else "tv"
    
    url = f"{TMDB_BASE_URL}/{endpoint_type}/{int(tmdb_id)}"
    params = {
        "api_key": TMDB_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            rating = data.get('vote_average')
            return rating if rating is not None else None
        else:
            print(f"Error fetching {content_type} ID {tmdb_id}: Status {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request error for {content_type} ID {tmdb_id}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error for {content_type} ID {tmdb_id}: {str(e)}")
        return None

def add_ratings_to_csv(
    input_file: str = "netflix_movies_tv_runtime.csv",
    output_file: str = "netflix_movies_tv_runtime_with_ratings.csv",
    batch_size: int = 100,
    delay: float = 0.25
):
    """
    Read CSV, fetch ratings from TMDB, and save updated CSV
    
    Args:
        input_file: Input CSV file path
        output_file: Output CSV file path
        batch_size: Save progress every N rows
        delay: Delay between API calls in seconds (to respect rate limits)
    """
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Add rating column if it doesn't exist
    if 'rating' not in df.columns:
        df['rating'] = None
    
    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    
    # Process each row
    for idx, row in df.iterrows():
        # Skip if rating already exists
        if pd.notna(df.at[idx, 'rating']):
            continue
        
        tmdb_id = row['tmdb_id']
        content_type = row['type']
        title = row['title']
        
        if pd.notna(tmdb_id):
            rating = get_tmdb_rating(tmdb_id, content_type)
            df.at[idx, 'rating'] = rating
            
            if rating is not None:
                print(f"[{idx+1}/{total_rows}] {title}: Rating = {rating}")
            else:
                print(f"[{idx+1}/{total_rows}] {title}: No rating found")
        else:
            print(f"[{idx+1}/{total_rows}] {title}: No TMDB ID")
        
        # Save progress every batch_size rows
        if (idx + 1) % batch_size == 0:
            print(f"\nSaving progress at row {idx+1}...")
            df.to_csv(output_file, index=False)
            print(f"Progress saved to {output_file}\n")
        
        # Delay to respect TMDB API rate limits (40 requests per 10 seconds)
        time.sleep(delay)
    
    # Final save
    print(f"\nSaving final results to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Print statistics
    total_with_rating = df['rating'].notna().sum()
    total_with_tmdb_id = df['tmdb_id'].notna().sum()
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Total rows: {total_rows}")
    print(f"Rows with TMDB ID: {total_with_tmdb_id}")
    print(f"Ratings fetched: {total_with_rating}")
    print(f"Success rate: {(total_with_rating/total_with_tmdb_id*100):.2f}%")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Check if API key is set
    if TMDB_API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please set your TMDB API key in the script!")
        print("Get your API key from: https://www.themoviedb.org/settings/api")
        exit(1)
    
    # Run the script
    add_ratings_to_csv(
        input_file="netflix_movies_tv_runtime.csv",
        output_file="netflix_movies_tv_runtime_with_ratings.csv",
        batch_size=100,  # Save progress every 100 rows
        delay=0.25  # 250ms delay = max 4 requests/second (safe for TMDB rate limit)
    )
