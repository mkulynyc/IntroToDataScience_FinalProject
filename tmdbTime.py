#!/usr/bin/env python3

import os
import pandas as pd
import requests
import time

def load_env_file():
    """Load environment variables from .env file"""
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ Loaded environment variables from .env file")
    else:
        print("⚠️  No .env file found")

def main():
    # Load environment variables from .env file
    load_env_file()
    
    # try to load a cleaned netflix dataset saved earlier
    csv_candidates = ['netflix_titles_cleaned.csv', 'netflix_titles.csv']
    df = None
    for c in csv_candidates:
        if os.path.exists(c):
            df = pd.read_csv(c)
            print(f"Loaded dataset: {c}")
            break

    if df is None:
        print("No Netflix CSV found (checked: {}). Place your dataset in the working directory and try again.".format(csv_candidates))
        return

    # Check if TMDB_API_KEY is set
    api_key = os.getenv('TMDB_API_KEY')
    
    if not api_key or api_key == 'your_api_key_here':
        print("❌ TMDB_API_KEY not found!")
        print("📝 Please add your API key to the .env file:")
        print("   1. Open the .env file in this directory")
        print("   2. Replace 'your_api_key_here' with your actual TMDB API key")
        print("   3. Save the file and run this script again")
        print("\n🔗 Get your free API key from: https://www.themoviedb.org/settings/api")
        return

    print(f"Found {len(df)} total records in dataset")
    
    # Get both movies and TV shows
    movies = df[df['type'].str.lower() == 'movie'].copy()
    tv_shows = df[df['type'].str.lower() == 'tv show'].copy()
    
    print(f"Found {len(movies)} movies to process")
    print(f"Found {len(tv_shows)} TV shows to process")
    print(f"Total content to process: {len(movies) + len(tv_shows)}")
    
    results = []
    
    # API URLs for different content types
    movie_search_url = "https://api.themoviedb.org/3/search/movie"
    tv_search_url = "https://api.themoviedb.org/3/search/tv"
    movie_details_url = "https://api.themoviedb.org/3/movie/{}"
    tv_details_url = "https://api.themoviedb.org/3/tv/{}"
    
    # Process Movies
    print("\n🎬 Processing Movies...")
    movie_titles = movies['title'].dropna().unique()
    total_items = len(movie_titles) + len(tv_shows['title'].dropna().unique())
    current_count = 0
    
    for title in movie_titles:
        current_count += 1
        try:
            print(f"Processing {current_count}/{total_items}: [MOVIE] {title}")
            
            # Search for movie
            r = requests.get(movie_search_url, params={'api_key': api_key, 'query': title, 'language': 'en-US'}, timeout=10)
            r.raise_for_status()
            res = r.json().get('results', [])
            
            if res:
                tmdb_id = res[0]['id']
                # Get movie details
                r2 = requests.get(movie_details_url.format(tmdb_id), params={'api_key': api_key, 'language': 'en-US'}, timeout=10)
                r2.raise_for_status()
                info = r2.json()
                
                # Get movie runtime in minutes
                movie_runtime = info.get('runtime')  # Already in minutes
                # Get movie rating
                rating = info.get('vote_average')
                
                results.append({
                    'title': title,
                    'type': 'Movie',
                    'tmdb_id': tmdb_id,
                    'movie_runtime_minutes': movie_runtime,
                    'episodes_total': None,
                    'total_seasons': None,
                    'episode_run_time': None,
                    'tv_minutes_total': None,
                    'rating': rating
                })
                
                if movie_runtime and rating:
                    print(f"  ✓ Runtime: {movie_runtime} minutes ({movie_runtime/60:.1f} hours) | Rating: {rating}/10")
                elif movie_runtime:
                    print(f"  ✓ Runtime: {movie_runtime} minutes ({movie_runtime/60:.1f} hours) | No rating")
            else:
                results.append({
                    'title': title,
                    'type': 'Movie',
                    'tmdb_id': None,
                    'movie_runtime_minutes': None,
                    'episodes_total': None,
                    'total_seasons': None,
                    'episode_run_time': None,
                    'tv_minutes_total': None,
                    'rating': None
                })
                print(f"  ✗ Not found on TMDB")
        except Exception as e:
            results.append({
                'title': title,
                'type': 'Movie',
                'tmdb_id': None,
                'movie_runtime_minutes': None,
                'episodes_total': None,
                'total_seasons': None,
                'episode_run_time': None,
                'tv_minutes_total': None,
                'rating': None
            })
            print(f"  ✗ Error: {str(e)}")
            
        time.sleep(0.25)  # be polite with API rate limits
    
    # Process TV Shows
    print("\n📺 Processing TV Shows...")
    tv_titles = tv_shows['title'].dropna().unique()
    
    for title in tv_titles:
        current_count += 1
        try:
            print(f"Processing {current_count}/{total_items}: [TV SHOW] {title}")
            
            # Search for TV show
            r = requests.get(tv_search_url, params={'api_key': api_key, 'query': title, 'language': 'en-US'}, timeout=10)
            r.raise_for_status()
            res = r.json().get('results', [])
            
            if res:
                tmdb_id = res[0]['id']
                # Get TV show details
                r2 = requests.get(tv_details_url.format(tmdb_id), params={'api_key': api_key, 'language': 'en-US'}, timeout=10)
                r2.raise_for_status()
                info = r2.json()
                
                # Get TV show data
                # Get TV show data
                runtimes = info.get('episode_run_time') or []
                episode_run_time = int(runtimes[0]) if runtimes else None
                episodes_total = info.get('number_of_episodes')
                total_seasons = info.get('number_of_seasons')
                # Calculate total TV minutes
                if episode_run_time and episodes_total:
                    tv_minutes_total = episodes_total * episode_run_time
                
                # Get TV show rating
                rating = info.get('vote_average')
                
                results.append({
                    'title': title,
                    'type': 'TV Show',
                    'tmdb_id': tmdb_id,
                    'movie_runtime_minutes': None,
                    'episodes_total': episodes_total,
                    'total_seasons': total_seasons,
                    'episode_run_time': episode_run_time,
                    'tv_minutes_total': tv_minutes_total,
                    'rating': rating
                })
                
                rating_text = f"Rating: {rating}/10" if rating else "No rating"
                if episodes_total and episode_run_time and total_seasons:
                    print(f"  ✓ {total_seasons} seasons, {episodes_total} episodes × {episode_run_time} min = {tv_minutes_total} total minutes ({tv_minutes_total/60:.1f} hours) | {rating_text}")
                elif episodes_total and total_seasons:
                    print(f"  ✓ {total_seasons} seasons, {episodes_total} episodes (no runtime data) | {rating_text}")
                elif episode_run_time:
                    print(f"  ✓ {episode_run_time} min per episode (no episode/season count) | {rating_text}")
                else:
                    print(f"  ✓ Found but no runtime data | {rating_text}")
            else:
                results.append({
                    'title': title,
                    'type': 'TV Show',
                    'tmdb_id': None,
                    'movie_runtime_minutes': None,
                    'episodes_total': None,
                    'total_seasons': None,
                    'episode_run_time': None,
                    'tv_minutes_total': None,
                    'rating': None
                })
                print(f"  ✗ Not found on TMDB")
        except Exception as e:
            results.append({
                'title': title,
                'type': 'TV Show',
                'tmdb_id': None,
                'movie_runtime_minutes': None,
                'episodes_total': None,
                'total_seasons': None,
                'episode_run_time': None,
                'tv_minutes_total': None,
                'rating': None
            })
            print(f"  ✗ Error: {str(e)}")
            
        time.sleep(0.25)  # be polite with API rate limits

    # Create DataFrame and save
    content_df = pd.DataFrame(results)
    output_file = 'netflix_movies_tv_runtime.csv'
    content_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    total_processed = len(content_df)
    movies_processed = len(content_df[content_df['type'] == 'Movie'])
    tv_shows_processed = len(content_df[content_df['type'] == 'TV Show'])
    
    # Movie statistics
    movies_with_runtime = len(content_df[(content_df['type'] == 'Movie') & (content_df['movie_runtime_minutes'].notna())])
    
    # TV show statistics
    tv_with_episodes = len(content_df[(content_df['type'] == 'TV Show') & (content_df['episodes_total'].notna())])
    tv_with_seasons = len(content_df[(content_df['type'] == 'TV Show') & (content_df['total_seasons'].notna())])
    tv_with_runtime = len(content_df[(content_df['type'] == 'TV Show') & (content_df['episode_run_time'].notna())])
    tv_with_total_minutes = len(content_df[(content_df['type'] == 'TV Show') & (content_df['tv_minutes_total'].notna())])
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Total content processed: {total_processed}")
    print(f"🎬 Movies processed: {movies_processed} (runtime data: {movies_with_runtime})")
    print(f"📺 TV shows processed: {tv_shows_processed}")
    print(f"   - Episode counts: {tv_with_episodes}")
    print(f"   - Season counts: {tv_with_seasons}")
    print(f"   - Episode runtime: {tv_with_runtime}")
    print(f"   - Total minutes calculated: {tv_with_total_minutes}")
    print(f"💾 Saved to: {output_file}")
    
    # Show sample data
    if len(content_df) > 0:
        print(f"\n📋 Sample of extracted data:")
        print(content_df.head())
        
        # Movie statistics
        movies_data = content_df[(content_df['type'] == 'Movie') & (content_df['movie_runtime_minutes'].notna())]
        if len(movies_data) > 0:
            print(f"\n🎬 Movie Statistics:")
            print(f"   Average movie runtime: {movies_data['movie_runtime_minutes'].mean():.1f} minutes ({movies_data['movie_runtime_minutes'].mean()/60:.1f} hours)")
            print(f"   Longest movie: {movies_data['movie_runtime_minutes'].max():.0f} minutes ({movies_data['movie_runtime_minutes'].max()/60:.1f} hours)")
            print(f"   Shortest movie: {movies_data['movie_runtime_minutes'].min():.0f} minutes ({movies_data['movie_runtime_minutes'].min()/60:.1f} hours)")
        
        # TV show statistics
        tv_data = content_df[(content_df['type'] == 'TV Show') & (content_df['tv_minutes_total'].notna())]
        if len(tv_data) > 0:
            print(f"\n� TV Show Statistics:")
            print(f"   Average seasons per show: {tv_data['total_seasons'].mean():.1f}")
            print(f"   Average episodes per show: {tv_data['episodes_total'].mean():.1f}")
            print(f"   Average episode runtime: {tv_data['episode_run_time'].mean():.1f} minutes")
            print(f"   Average total runtime per show: {tv_data['tv_minutes_total'].mean():.0f} minutes ({tv_data['tv_minutes_total'].mean()/60:.1f} hours)")
            print(f"   Longest show: {tv_data['tv_minutes_total'].max():.0f} minutes ({tv_data['tv_minutes_total'].max()/60:.1f} hours)")
            print(f"   Shortest show: {tv_data['tv_minutes_total'].min():.0f} minutes ({tv_data['tv_minutes_total'].min()/60:.1f} hours)")

if __name__ == '__main__':
    main()

