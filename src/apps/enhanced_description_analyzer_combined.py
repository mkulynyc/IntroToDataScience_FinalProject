
"""
🎭 Enhanced Netflix Description Analyzer - All-in-One
====================================================

Combined script that provides both:
1. Command-line analysis engine (adds NLP features to Netflix data)
2. Interactive Streamlit web app (explores analysis results)

Features:
- Emotion analysis (sentiment polarity, subjectivity)
- Readability analysis (Flesch Reading Ease scores)
- AI summarization (BART transformer)
- TF-IDF keyword extraction
- Word count analysis
- Interactive visualizations and filtering

Usage:
- Command line: python enhanced_description_analyzer.py
- Web app: streamlit run enhanced_description_analyzer.py

Requirements:
- streamlit, pandas, textblob, textstat, transformers, scikit-learn
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import textstat
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import sys
import os

# Initialize transformers pipeline for summarization (lazy loading)
summarizer = None
TRANSFORMERS_AVAILABLE = False

def check_transformers_availability():
    """Check if transformers is available without importing it"""
    global TRANSFORMERS_AVAILABLE
    if not TRANSFORMERS_AVAILABLE:
        try:
            import transformers
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
    return TRANSFORMERS_AVAILABLE

def get_summarizer():
    """Lazy load the summarizer to avoid memory issues at import time"""
    global summarizer
    if summarizer is None and check_transformers_availability():
        try:
            from transformers import pipeline
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            print(f"Warning: Could not load summarizer model: {e}")
            summarizer = None
    return summarizer

def analyze_emotion(text):
    """
    Analyze emotion/sentiment of text using TextBlob
    Returns: dict with polarity, subjectivity, and overall sentiment
    """
    if not text or pd.isna(text):
        return {
            'polarity': 0.0,
            'subjectivity': 0.0,
            'sentiment': 'neutral'
        }

    blob = TextBlob(text)

    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Classify overall sentiment
    if polarity > 0.1:
        sentiment = 'positive'
    elif polarity < -0.1:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return {
        'polarity': round(polarity, 3),
        'subjectivity': round(subjectivity, 3),
        'sentiment': sentiment
    }

def analyze_readability(text):
    """
    Analyze readability of text using textstat
    Returns: dict with score and category
    """
    if not text or pd.isna(text):
        return {
            'score': 0.0,
            'category': 'unknown'
        }

    # Calculate Flesch Reading Ease score
    score = textstat.flesch_reading_ease(text)

    # Categorize readability
    if score >= 90:
        category = 'very_easy'
    elif score >= 80:
        category = 'easy'
    elif score >= 70:
        category = 'fairly_easy'
    elif score >= 60:
        category = 'standard'
    elif score >= 50:
        category = 'fairly_difficult'
    elif score >= 30:
        category = 'difficult'
    else:
        category = 'very_difficult'

    return {
        'score': round(score, 1),
        'category': category
    }

def generate_ai_summary(text):
    """
    Generate AI summary using transformers (BART)
    Returns: short summary string
    """
    if not text or pd.isna(text):
        return text[:100] + "..." if len(str(text)) > 100 else str(text)

    summarizer = get_summarizer()
    if not summarizer:
        return text[:100] + "..." if len(str(text)) > 100 else str(text)

    try:
        # Only summarize longer texts
        text_str = str(text)
        if len(text_str.split()) < 20:
            return text_str

        # Generate summary
        summary = summarizer(text_str, max_length=50, min_length=10, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Warning: Could not generate summary: {e}")
        return text_str[:100] + "..." if len(text_str) > 100 else text_str

def extract_global_keywords(descriptions, max_features=15):
    """
    Extract global TF-IDF keywords from all descriptions
    Returns: list of top keywords
    """
    if not descriptions:
        return []

    # Clean and preprocess descriptions
    processed_texts = []
    for desc in descriptions:
        if desc and not pd.isna(desc):
            # Remove special characters and extra whitespace
            text = re.sub(r'[^\w\s]', ' ', str(desc))
            text = re.sub(r'\s+', ' ', text).strip().lower()
            processed_texts.append(text)

    if not processed_texts:
        return []

    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(processed_texts)

        # Get feature names (keywords)
        keywords = vectorizer.get_feature_names_out()
        return list(keywords)
    except Exception as e:
        print(f"Warning: Could not extract keywords: {e}")
        return []

def count_words(text):
    """
    Count number of words in text
    Returns: integer word count
    """
    if not text or pd.isna(text):
        return 0

    return len(str(text).split())

def enrich_netflix_with_enhanced_description_analysis(netflix_df):
    """
    Add emotion, readability, and advanced NLP analysis to Netflix dataframe
    Only adds columns that don't already exist
    """
    print("🔍 Analyzing descriptions for emotion, readability, and advanced NLP features...")

    # Check which columns already exist
    existing_columns = set(netflix_df.columns)
    emotion_cols = {'description_emotion_polarity', 'description_emotion_subjectivity', 'description_emotion_sentiment'}
    readability_cols = {'description_readability_score', 'description_readability_category'}
    nlp_cols = {'nlp_summary', 'nlp_global_keywords', 'nlp_word_count'}

    # Add emotion analysis if not already present
    if not emotion_cols.issubset(existing_columns):
        print("😊 Adding emotion analysis...")
        emotion_data = []
        for desc in netflix_df['description']:
            emotion_data.append(analyze_emotion(desc))

        # Convert to dataframe and merge
        emotion_df = pd.DataFrame(emotion_data)
        emotion_df.columns = ['description_emotion_' + col for col in emotion_df.columns]
    else:
        emotion_df = netflix_df[['description_emotion_polarity', 'description_emotion_subjectivity', 'description_emotion_sentiment']].copy()
        emotion_df.columns = ['description_emotion_' + col.split('_')[-1] for col in emotion_df.columns]

    # Add readability analysis if not already present
    if not readability_cols.issubset(existing_columns):
        print("📚 Adding readability analysis...")
        readability_data = []
        for desc in netflix_df['description']:
            readability_data.append(analyze_readability(desc))

        # Convert to dataframe and merge
        readability_df = pd.DataFrame(readability_data)
        readability_df.columns = ['description_readability_' + col for col in readability_df.columns]
    else:
        readability_df = netflix_df[['description_readability_score', 'description_readability_category']].copy()
        readability_df.columns = ['description_readability_' + col.split('_')[-1] for col in readability_df.columns]

    # Add advanced NLP features (only if not already present)
    if 'nlp_summary' not in existing_columns:
        print("🤖 Generating AI summaries...")
        netflix_df['nlp_summary'] = [generate_ai_summary(desc) for desc in netflix_df['description']]
    else:
        print("🤖 AI summaries already present, skipping...")

    if 'nlp_global_keywords' not in existing_columns:
        print("🔑 Extracting global TF-IDF keywords...")
        descriptions = netflix_df['description'].tolist()
        global_keywords = extract_global_keywords(descriptions, max_features=15)
        netflix_df['nlp_global_keywords'] = [json.dumps(global_keywords)] * len(netflix_df)
    else:
        print("🔑 Global keywords already present, skipping...")

    if 'nlp_word_count' not in existing_columns:
        print("📊 Counting words...")
        netflix_df['nlp_word_count'] = [count_words(desc) for desc in netflix_df['description']]
    else:
        print("📊 Word counts already present, skipping...")

    # Combine all dataframes (only add emotion/readability if they weren't already present)
    if not emotion_cols.issubset(existing_columns) or not readability_cols.issubset(existing_columns):
        enriched_df = pd.concat([netflix_df, emotion_df, readability_df], axis=1)
    else:
        enriched_df = netflix_df.copy()

    return enriched_df

def run_command_line_analysis():
    """Run the enhanced analysis from command line"""
    print("🎭 Enhanced Netflix Description Analyzer")
    print("=" * 50)

    # File paths
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    input_file = data_dir / "netflix_with_description_analysis.csv"  # Use analyzed dataset
    output_file = data_dir / "netflix_with_enhanced_description_analysis.csv"

    # Load Netflix data
    print("Loading Netflix data...")
    try:
        netflix_df = pd.read_csv(input_file)
        print(f"Loaded pre-analyzed dataset with {len(netflix_df)} records")
    except FileNotFoundError:
        # Fallback to clean dataset if analyzed doesn't exist
        input_file = data_dir / "netflix_clean.csv"
        print(f"Pre-analyzed dataset not found, loading clean dataset...")
        netflix_df = pd.read_csv(input_file)

    print(f"Processing {len(netflix_df)} descriptions...")

    # Add enhanced analysis columns
    enriched_df = enrich_netflix_with_enhanced_description_analysis(netflix_df)

    # Save enriched data
    print(f"Saving enhanced data to {output_file}...")
    enriched_df.to_csv(output_file, index=False)

    print("✅ Enhanced analysis complete!")
    print(f"📊 Original columns: {len(netflix_df.columns)}")
    print(f"📊 Enhanced columns: {len(enriched_df.columns)}")
    print(f"📁 Output saved to: {output_file}")

    # Show sample results
    print("\n📈 Sample Enhanced Analysis Results:")
    sample_cols = ['title', 'description_emotion_sentiment', 'description_emotion_polarity',
                   'description_readability_category', 'nlp_word_count']
    sample = enriched_df.head(3)[sample_cols]
    print(sample.to_string(index=False))

    # Show sample NLP features
    print("\n🤖 Sample NLP Features:")
    nlp_sample = enriched_df.head(1)[['title', 'nlp_summary', 'nlp_global_keywords', 'nlp_word_count']]
    for _, row in nlp_sample.iterrows():
        print(f"Title: {row['title']}")
        print(f"AI Summary: {row['nlp_summary'][:100]}...")
        print(f"Word Count: {row['nlp_word_count']}")
        print(f"Global Keywords: {row['nlp_global_keywords']}")
        print()

    # Show new columns summary
    new_cols = [col for col in enriched_df.columns if col not in netflix_df.columns]
    print(f"\n🆕 Added {len(new_cols)} new enhanced analysis columns:")
    for col in new_cols:
        print(f"  • {col}")

def run_streamlit_app():
    """Run the Streamlit web application"""
    import streamlit as st
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    # Page configuration
    st.set_page_config(
        page_title="Enhanced Netflix Description Analyzer",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #E50914;
            text-align: center;
            margin-bottom: 2rem;
        }
        .emotion-positive {
            background-color: #d4edda;
            color: #155724;
            padding: 0.5rem;
            border-radius: 0.25rem;
            font-weight: bold;
        }
        .emotion-negative {
            background-color: #f8d7da;
            color: #721c24;
            padding: 0.5rem;
            border-radius: 0.25rem;
            font-weight: bold;
        }
        .emotion-neutral {
            background-color: #fff3cd;
            color: #856404;
            padding: 0.5rem;
            border-radius: 0.25rem;
            font-weight: bold;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 0.25rem solid #E50914;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown('<div class="main-header">🎭 Enhanced Netflix Description Analyzer</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("📊 Analysis Options")

    # File upload or use existing data
    data_option = st.sidebar.radio(
        "Choose Data Source:",
        ["Upload CSV", "Use Existing Netflix Data"]
    )

    df = None
    if data_option == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload Netflix CSV", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
    else:
        # Try to load existing data
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        enhanced_file = data_dir / "netflix_with_enhanced_description_analysis.csv"
        analyzed_file = data_dir / "netflix_with_description_analysis.csv"
        clean_file = data_dir / "netflix_clean.csv"

        for file_path in [enhanced_file, analyzed_file, clean_file]:
            if file_path.exists():
                df = pd.read_csv(file_path)
                dataset_type = "enhanced" if "enhanced" in file_path.name else ("analyzed" if "description_analysis" in file_path.name else "clean")
                st.sidebar.success(f"Loaded {len(df)} records from {dataset_type} dataset")
                st.sidebar.write(f"**File loaded:** {file_path.name}")
                break

        if df is None:
            st.sidebar.error("No Netflix data found. Please upload a CSV.")

    if df is not None and 'description' in df.columns:
        # Check if data already has enhanced analysis
        enhanced_cols = {'nlp_summary', 'nlp_word_count', 'description_emotion_polarity',
                        'description_emotion_subjectivity', 'description_emotion_sentiment',
                        'description_readability_score', 'description_readability_category',
                        'nlp_global_keywords'}

        has_enhanced_analysis = enhanced_cols.issubset(set(df.columns))

        # Debug information
        st.sidebar.write("---")
        st.sidebar.write("**Debug Info:**")
        st.sidebar.write(f"Total columns: {len(df.columns)}")
        st.sidebar.write(f"Enhanced analysis detected: {has_enhanced_analysis}")
        if not has_enhanced_analysis:
            missing = enhanced_cols - set(df.columns)
            st.sidebar.write(f"Missing columns: {list(missing)}")
            st.sidebar.write("**Available columns:**")
            st.sidebar.write(list(df.columns))

        if has_enhanced_analysis:
            st.info(f"📊 Loaded fully enhanced dataset with {len(df)} records and all analysis columns!")
            enriched_df = df.copy()
            st.session_state['enriched_df'] = enriched_df
            # Show results immediately since we have enhanced data
            show_results(enriched_df)
        else:
            # Analysis section
            st.header("🔍 Enhanced Description Analysis")

            if st.button("🚀 Run Enhanced Analysis", type="primary"):
                with st.spinner("Analyzing descriptions... This may take a few minutes for AI summarization."):
                    # Run the enhanced analysis
                    enriched_df = enrich_netflix_with_enhanced_description_analysis(df.copy())

                st.success("✅ Analysis Complete!")

                # Store in session state
                st.session_state['enriched_df'] = enriched_df
                # Show results after analysis
                show_results(enriched_df)

        # Display results if available (for cases where session state persists)
        if 'enriched_df' in st.session_state and not has_enhanced_analysis:
            enriched_df = st.session_state['enriched_df']
            show_results(enriched_df)

def show_results(enriched_df):
    """Display the analysis results"""
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Titles", len(enriched_df))
    with col2:
        pos_count = (enriched_df['description_emotion_sentiment'] == 'positive').sum()
        st.metric("Positive Descriptions", pos_count)
    with col3:
        avg_readability = enriched_df['description_readability_score'].mean()
        st.metric("Avg Readability", f"{avg_readability:.1f}")
    with col4:
        avg_words = enriched_df['nlp_word_count'].mean()
        st.metric("Avg Word Count", f"{avg_words:.0f}")

    # Emotion Analysis
    st.header("😊 Emotion Analysis")
    emotion_counts = enriched_df['description_emotion_sentiment'].value_counts()
    fig_emotion = px.pie(
        values=emotion_counts.values,
        names=emotion_counts.index,
        title="Description Sentiment Distribution",
        color_discrete_sequence=['#d4edda', '#f8d7da', '#fff3cd']
    )
    st.plotly_chart(fig_emotion, use_container_width=True)

    # Readability Analysis
    st.header("📖 Readability Analysis")
    readability_counts = enriched_df['description_readability_category'].value_counts()
    fig_readability = px.bar(
        x=readability_counts.index,
        y=readability_counts.values,
        title="Readability Categories",
        color=readability_counts.index,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig_readability, use_container_width=True)

    # Word Count Distribution
    st.header("📊 Word Count Analysis")
    fig_words = px.histogram(
        enriched_df,
        x='nlp_word_count',
        title="Distribution of Word Counts in Descriptions",
        nbins=30,
        color_discrete_sequence=['#E50914']
    )
    st.plotly_chart(fig_words, use_container_width=True)

    # Sample Results
    st.header("📋 Sample Analysis Results")
    sample_cols = [
        'title', 'type', 'description_emotion_sentiment',
        'description_emotion_polarity', 'description_readability_category',
        'nlp_word_count', 'nlp_summary'
    ]
    st.dataframe(enriched_df[sample_cols].head(10), use_container_width=True)

    # Global Keywords
    st.header("🔑 Global Keywords")
    if enriched_df['nlp_global_keywords'].notna().any():
        keywords_str = enriched_df['nlp_global_keywords'].iloc[0]
        try:
            keywords = json.loads(keywords_str)
            st.write("Top TF-IDF keywords across all descriptions:")
            st.write(", ".join(keywords))
        except:
            st.write("Keywords data not available")

    # Download Results
    st.header("💾 Download Enhanced Data")
    csv = enriched_df.to_csv(index=False)
    st.download_button(
        label="Download CSV with Enhanced Analysis",
        data=csv,
        file_name="netflix_enhanced_analysis.csv",
        mime="text/csv"
    )

# Single Description Analysis (always available)
st.header("🔍 Single Description Analysis")
user_description = st.text_area(
    "Enter a Netflix-style description to analyze:",
    height=100,
    placeholder="Enter a movie or TV show description here..."
)

if st.button("Analyze Description") and user_description.strip():
    with st.spinner("Analyzing..."):
        # Analyze single description
        emotion = analyze_emotion(user_description)
        readability = analyze_readability(user_description)
        summary = generate_ai_summary(user_description)
        word_count = count_words(user_description)

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Emotion Analysis")
        sentiment_class = f"emotion-{emotion['sentiment']}"
        st.markdown(f"<div class='{sentiment_class}'>{emotion['sentiment'].title()}</div>", unsafe_allow_html=True)
        st.write(f"**Polarity:** {emotion['polarity']}")
        st.write(f"**Subjectivity:** {emotion['subjectivity']}")

    with col2:
        st.subheader("Readability")
        st.write(f"**Score:** {readability['score']}")
        st.write(f"**Category:** {readability['category'].replace('_', ' ').title()}")

    st.subheader("AI Summary")
    st.write(summary)

    st.subheader("Word Count")
    st.write(f"{word_count} words")

# Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and advanced NLP libraries*")

def main():
    """Main entry point - detects how the script is being run"""
    try:
        import streamlit as st
        # If we can import streamlit, we're likely running in streamlit
        run_streamlit_app()
    except ImportError:
        # Streamlit not available, run command line analysis
        run_command_line_analysis()

if __name__ == "__main__":
    main()