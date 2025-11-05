"""
Enhanced Netflix Description Analyzer - Analysis Engine
=======================================================

Core analysis engine that adds NLP features to Netflix data.
"""

import pandas as pd
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import sys
import os

# Import shared utilities from parent directory
sys.path.append(str(Path(__file__).parent.parent))
from shared_utils import analyze_emotion, analyze_readability, count_words

# Initialize transformers pipeline for summarization
summarizer = None
TRANSFORMERS_AVAILABLE = False

def check_transformers_availability():
    """Check if transformers is available"""
    global TRANSFORMERS_AVAILABLE
    if not TRANSFORMERS_AVAILABLE:
        try:
            import transformers
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
    return TRANSFORMERS_AVAILABLE

def get_summarizer():
    """Lazy load the summarizer"""
    global summarizer
    if summarizer is None and check_transformers_availability():
        try:
            from transformers import pipeline
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            print(f"Warning: Could not load summarizer model: {e}")
            summarizer = None
    return summarizer

def generate_ai_summary(text):
    """Generate AI summary using transformers (BART)"""
    if not text or pd.isna(text):
        return text[:100] + "..." if len(str(text)) > 100 else str(text)

    summarizer = get_summarizer()
    if not summarizer:
        return text[:100] + "..." if len(str(text)) > 100 else str(text)

    try:
        text_str = str(text)
        if len(text_str.split()) < 20:
            return text_str

        summary = summarizer(text_str, max_length=50, min_length=10, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Warning: Could not generate summary: {e}")
        return text_str[:100] + "..." if len(text_str) > 100 else text_str

def extract_global_keywords(descriptions, max_features=15):
    """Extract global TF-IDF keywords"""
    if not descriptions:
        return []

    processed_texts = []
    for desc in descriptions:
        if desc and not pd.isna(desc):
            text = re.sub(r'[^\w\s]', ' ', str(desc))
            text = re.sub(r'\s+', ' ', text).strip().lower()
            processed_texts.append(text)

    if not processed_texts:
        return []

    try:
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        keywords = vectorizer.get_feature_names_out()
        return list(keywords)
    except Exception as e:
        print(f"Warning: Could not extract keywords: {e}")
        return []

def enrich_netflix_with_enhanced_description_analysis(netflix_df):
    """Add emotion, readability, and advanced NLP analysis"""
    print("Analyzing descriptions for emotion, readability, and advanced NLP features...")

    existing_columns = set(netflix_df.columns)
    emotion_cols = {'description_emotion_polarity', 'description_emotion_subjectivity', 'description_emotion_sentiment'}
    readability_cols = {'description_readability_score', 'description_readability_category'}

    if not emotion_cols.issubset(existing_columns):
        print("Adding emotion analysis...")
        emotion_data = []
        for desc in netflix_df['description']:
            emotion_data.append(analyze_emotion(desc))
        emotion_df = pd.DataFrame(emotion_data)
        emotion_df.columns = ['description_emotion_' + col for col in emotion_df.columns]
    else:
        emotion_df = netflix_df[list(emotion_cols)].copy()

    if not readability_cols.issubset(existing_columns):
        print("Adding readability analysis...")
        readability_data = []
        for desc in netflix_df['description']:
            readability_data.append(analyze_readability(desc))
        readability_df = pd.DataFrame(readability_data)
        readability_df.columns = ['description_readability_' + col for col in readability_df.columns]
    else:
        readability_df = netflix_df[list(readability_cols)].copy()

    if 'nlp_summary' not in existing_columns:
        print("Generating AI summaries...")
        netflix_df['nlp_summary'] = [generate_ai_summary(desc) for desc in netflix_df['description']]

    if 'nlp_global_keywords' not in existing_columns:
        print("Extracting global TF-IDF keywords...")
        descriptions = netflix_df['description'].tolist()
        global_keywords = extract_global_keywords(descriptions, max_features=15)
        netflix_df['nlp_global_keywords'] = [json.dumps(global_keywords)] * len(netflix_df)

    if 'nlp_word_count' not in existing_columns:
        print("Counting words...")
        netflix_df['nlp_word_count'] = [count_words(desc) for desc in netflix_df['description']]

    if not emotion_cols.issubset(existing_columns) or not readability_cols.issubset(existing_columns):
        enriched_df = pd.concat([netflix_df, emotion_df, readability_df], axis=1)
    else:
        enriched_df = netflix_df.copy()

    return enriched_df

def main():
    """Main entry point"""
    print("Enhanced Netflix Description Analyzer")
    print("=" * 50)

    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    input_file = data_dir / "netflix_with_description_analysis.csv"
    output_file = data_dir / "netflix_with_enhanced_description_analysis.csv"

    print("Loading Netflix data...")
    try:
        netflix_df = pd.read_csv(input_file)
        print(f"Loaded pre-analyzed dataset with {len(netflix_df)} records")
    except FileNotFoundError:
        input_file = data_dir / "netflix_clean.csv"
        print(f"Pre-analyzed dataset not found, loading clean dataset...")
        netflix_df = pd.read_csv(input_file)

    print(f"Processing {len(netflix_df)} descriptions...")
    enriched_df = enrich_netflix_with_enhanced_description_analysis(netflix_df)

    print(f"Saving enhanced data to {output_file}...")
    enriched_df.to_csv(output_file, index=False)

    print("Enhanced analysis complete!")
    print(f"Original columns: {len(netflix_df.columns)}")
    print(f"Enhanced columns: {len(enriched_df.columns)}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()
