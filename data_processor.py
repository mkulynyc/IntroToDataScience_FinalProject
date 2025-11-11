"""
Data Processing Script for Netflix Recommender
This script processes the Netflix data and saves it to pickle files for faster loading in Streamlit
"""

import pandas as pd
import pickle
import os
from recomender_v1 import NetflixAnalyticsPipeline

def process_and_save_data():
    """Process Netflix data and save to pickle files"""
    
    # Create pickle directory
    pickle_dir = "pickle_data"
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
    
    # Check if data files exist
    netflix_file = "netflix_titles.csv"
    runtime_file = "netflix_movies_tv_runtime.csv"
    
    if not os.path.exists(netflix_file):
        print(f"Error: {netflix_file} not found!")
        print("Please ensure the Netflix dataset is in the current directory.")
        return False
    
    print("Processing Netflix data...")
    print("This may take several minutes...")
    
    try:
        # Initialize and run pipeline
        pipeline = NetflixAnalyticsPipeline()
        
        # Use runtime file if it exists
        runtime_path = runtime_file if os.path.exists(runtime_file) else None
        results = pipeline.run_full_pipeline(netflix_file, runtime_path)
        
        # Extract data
        df = results['dataframe']
        
        # Save pipeline and data to pickle
        pipeline_pickle = os.path.join(pickle_dir, "netflix_pipeline.pkl")
        data_pickle = os.path.join(pickle_dir, "netflix_data.pkl")
        
        with open(pipeline_pickle, 'wb') as f:
            pickle.dump(pipeline, f)
        
        with open(data_pickle, 'wb') as f:
            pickle.dump(df, f)
        
        print(f"✅ Data processed successfully!")
        print(f"✅ Pipeline saved to: {pipeline_pickle}")
        print(f"✅ Data saved to: {data_pickle}")
        print(f"📊 Total records: {len(df)}")
        type_col = 'type' if 'type' in df.columns else 'type_x' if 'type_x' in df.columns else None
        if type_col:
            print(f"🎬 Movies: {len(df[df[type_col] == 'Movie'])}")
            print(f"📺 TV Shows: {len(df[df[type_col] == 'TV Show'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing data: {str(e)}")
        return False

def check_pickle_files():
    """Check if pickle files exist and show their info"""
    pickle_dir = "pickle_data"
    pipeline_pickle = os.path.join(pickle_dir, "netflix_pipeline.pkl")
    data_pickle = os.path.join(pickle_dir, "netflix_data.pkl")
    
    print("Checking pickle files...")
    
    if os.path.exists(pipeline_pickle):
        size = os.path.getsize(pipeline_pickle) / (1024 * 1024)  # MB
        print(f"✅ Pipeline file exists: {pipeline_pickle} ({size:.2f} MB)")
    else:
        print(f"❌ Pipeline file missing: {pipeline_pickle}")
    
    if os.path.exists(data_pickle):
        size = os.path.getsize(data_pickle) / (1024 * 1024)  # MB
        print(f"✅ Data file exists: {data_pickle} ({size:.2f} MB)")
        
        # Load and show basic info
        try:
            with open(data_pickle, 'rb') as f:
                df = pickle.load(f)
            print(f"📊 Records in pickle: {len(df)}")
            print(f"📅 Date range: {df['release_year'].min()} - {df['release_year'].max()}")
        except Exception as e:
            print(f"⚠️ Error reading data pickle: {str(e)}")
    else:
        print(f"❌ Data file missing: {data_pickle}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_pickle_files()
    else:
        print("Netflix Recommender Data Processor")
        print("==================================")
        
        # Check current files
        check_pickle_files()
        
        # Ask user if they want to process
        print("\nOptions:")
        print("1. Process data (this will take several minutes)")
        print("2. Check existing files only")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            success = process_and_save_data()
            if success:
                print("\n🚀 Ready to run Streamlit app!")
                print("Run: streamlit run app.py")
        elif choice == "2":
            print("File check completed.")
        else:
            print("Invalid choice.")
