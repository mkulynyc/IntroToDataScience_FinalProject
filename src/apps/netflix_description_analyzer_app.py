#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Netflix Description Analyzer
===========================

A simple tool to analyze Netflix show descriptions for sentiment and readability.
Created for exploring patterns in Netflix content descriptions.

Author: Netflix Data Analysis Team
Last modified: November 2024
"""

import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from pathlib import Path
import time
import warnings
import traceback

# Import shared utilities to avoid code duplication
import sys
from pathlib import Path
current_dir = Path(__file__).parent

parent_dir = current_dir.parent
sys.path.append(str(parent_dir / "utils"))

from src.utils.shared_utils import (
    analyze_emotion, 
    analyze_readability, 
    load_netflix_data,
    get_streamlit_color_classes,
    DEFAULT_PORTS,
    NETFLIX_COLORS
)

warnings.filterwarnings('ignore')

def main():
    """Run the Streamlit web application"""

    # Page configuration
    st.set_page_config(
        page_title="Netflix Description Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply shared CSS styles for consistent appearance across apps
    st.markdown(f"<style>{get_streamlit_color_classes()}</style>", unsafe_allow_html=True)

    # Header section
    st.markdown('<div class="main-header">Netflix Description Analyzer</div>', unsafe_allow_html=True)
    st.markdown("*Simple tool for analyzing sentiment and readability of Netflix show descriptions*")
    
    # Quick explanation for users
    with st.expander("What does this tool do?"):
        st.write("""
        This analyzer looks at Netflix show descriptions and calculates:
        - **Sentiment**: How positive/negative the description sounds
        - **Readability**: How easy the description is to read (grade level)
        
        It's pretty basic but gives you a quick overview of the data.
        """)

    # Sidebar controls
    st.sidebar.title("Options")

    # Data source selection - keeping it simple
    data_option = st.sidebar.radio(
        "Data Source:",
        ["Use Netflix Dataset", "Upload Your Own CSV"]
    )

    # Load data - try a few different approaches since path handling can be tricky
    df = None
    
    if data_option == "Upload Your Own CSV":
        uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"Loaded {len(df)} rows")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {str(e)}")
    else:
        # Look for Netflix data in the expected locations
        # This is a bit hacky but works regardless of where the script is run from
        possible_paths = [
            Path(__file__).parent.parent / "data" / "netflix_with_description_analysis.csv",  # Analyzed version first
            Path(__file__).parent.parent / "data" / "netflix_clean.csv",  # Fallback to clean version
            Path("data/netflix_with_description_analysis.csv"),  # Relative paths
            Path("data/netflix_clean.csv"),
            Path("../data/netflix_clean.csv"),  # In case we're running from a subfolder
        ]

        data_file = None
        for path in possible_paths:
            if path.exists():
                data_file = path
                break

        if data_file:
            try:
                df = pd.read_csv(data_file)
                dataset_type = "pre-analyzed" if "analysis" in data_file.name else "raw"
                st.sidebar.success(f"Found Netflix data: {len(df)} shows ({dataset_type})")
                st.sidebar.info(f"Using: {data_file.name}")
            except Exception as e:
                st.sidebar.error(f"Error loading Netflix data: {str(e)}")
        else:
            st.sidebar.warning("No Netflix data found. Please upload a CSV file.")
            st.sidebar.info("Expected files: netflix_clean.csv or netflix_with_description_analysis.csv")

    if df is not None and 'title' in df.columns:
        # Check if data already has analysis
        analysis_cols = {'description_emotion_polarity', 'description_readability_score'}
        has_analysis = analysis_cols.issubset(set(df.columns))

        if has_analysis:
            st.info(f"Loaded pre-analyzed dataset: {len(df)} records")
            analyzed_df = df.copy()
        else:
            # Analysis section
            st.header("Run Description Analysis")
            st.write("This will add sentiment and readability scores to your Netflix data.")
            
            # Give user some control over the process
            sample_size = st.slider("Sample size (for testing)", 10, min(1000, len(df)), 
                                   value=min(100, len(df)), step=10)
            
            use_sample = st.checkbox("Use sample for faster processing", value=True)
            
            if st.button("Start Analysis", type="primary"):
                # Decide whether to use full dataset or sample
                work_df = df.sample(n=sample_size) if use_sample else df.copy()
                
                with st.spinner(f"Analyzing {len(work_df)} Netflix descriptions... Grab a coffee ☕"):
                    analyzed_df = work_df.copy()
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_rows = len(work_df)
                    
                    # Set up new columns - doing it this way is more explicit
                    analyzed_df['description_emotion_polarity'] = 0.0
                    analyzed_df['description_emotion_subjectivity'] = 0.0
                    analyzed_df['description_emotion_sentiment'] = 'neutral'
                    analyzed_df['description_readability_score'] = 0.0
                    analyzed_df['description_readability_category'] = 'unknown'
                    
                    # Process each description
                    # Using iterrows() isn't the fastest but it's readable and works fine for our dataset size
                    for idx, row in work_df.iterrows():
                        # Update progress - users like to see things happening
                        progress = ((idx - work_df.index[0]) + 1) / total_rows
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {row['title'][:50]}..." if len(row['title']) > 50 else f"Processing: {row['title']}")
                        
                        # Run the actual analysis
                        if pd.notna(row['description']):  # Skip null descriptions
                            polarity, subjectivity, sentiment = analyze_emotion(row['description'])
                            analyzed_df.loc[idx, 'description_emotion_polarity'] = polarity
                            analyzed_df.loc[idx, 'description_emotion_subjectivity'] = subjectivity
                            analyzed_df.loc[idx, 'description_emotion_sentiment'] = sentiment
                            
                            readability_score, readability_category = analyze_readability(row['description'])
                            analyzed_df.loc[idx, 'description_readability_score'] = readability_score
                            analyzed_df.loc[idx, 'description_readability_category'] = readability_category
                        
                        # Brief pause every so often so the UI stays responsive
                        if idx % 50 == 0:
                            time.sleep(0.1)
                    
                    progress_bar.progress(1.0)
                    status_text.text("Analysis Complete!")

                st.success("Analysis Complete!")
                
                # Save results in session state so user doesn't lose work if they refresh
                st.session_state['analyzed_df'] = analyzed_df
                
                # Quick debug info for development
                if st.checkbox("Show debug info", value=False):
                    st.write("Debug: Analysis results shape:", analyzed_df.shape)
                    st.write("Debug: New columns added:", [col for col in analyzed_df.columns if 'description_' in col])

        # Show results if we have them (either just calculated or from session state)
        if 'analyzed_df' in st.session_state or 'analyzed_df' in locals():
            if 'analyzed_df' not in locals():
                analyzed_df = st.session_state['analyzed_df']  # Restore from session

            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_polarity = analyzed_df['description_emotion_polarity'].mean()
                st.metric("Average Sentiment", f"{avg_polarity:.3f}")
            with col2:
                positive_count = (analyzed_df['description_emotion_sentiment'] == 'positive').sum()
                st.metric("Positive Descriptions", positive_count)
            with col3:
                avg_readability = analyzed_df['description_readability_score'].mean()
                st.metric("Average Readability", f"{avg_readability:.1f}")
            with col4:
                st.metric("Total Analyzed", len(analyzed_df))

            # Sentiment Distribution
            st.header("Analysis Results")
            sentiment_counts = analyzed_df['description_emotion_sentiment'].value_counts()
            fig_sentiment = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution of Netflix Descriptions"
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)

            # Readability Distribution
            fig_readability = px.histogram(
                analyzed_df,
                x='description_readability_score',
                title="Distribution of Readability Scores",
                nbins=20
            )
            st.plotly_chart(fig_readability, use_container_width=True)

            # Sentiment by Content Type
            if 'type' in analyzed_df.columns:
                sentiment_by_type = pd.crosstab(
                    analyzed_df['type'], 
                    analyzed_df['description_emotion_sentiment'],
                    normalize='index'
                ) * 100

                fig_type_sentiment = px.bar(
                    sentiment_by_type,
                    title="Sentiment Distribution by Content Type (%)",
                    barmode='stack'
                )
                st.plotly_chart(fig_type_sentiment, use_container_width=True)

            # Sample Results
            st.subheader("Sample Analysis Results")
            
            # Positive examples
            st.write("**Most Positive Descriptions:**")
            positive_samples = analyzed_df.nlargest(3, 'description_emotion_polarity')
            for _, row in positive_samples.iterrows():
                st.markdown(
                    f"<div class='emotion-positive'>{row['title']} - Sentiment: {row['description_emotion_polarity']:.3f}</div>",
                    unsafe_allow_html=True
                )

            # Negative examples
            st.write("**Most Negative Descriptions:**")
            negative_samples = analyzed_df.nsmallest(3, 'description_emotion_polarity')
            for _, row in negative_samples.iterrows():
                st.markdown(
                    f"<div class='emotion-negative'>{row['title']} - Sentiment: {row['description_emotion_polarity']:.3f}</div>",
                    unsafe_allow_html=True
                )

            # Full results table
            st.subheader("Complete Analysis Results")
            display_cols = [
                'title', 'type', 'description_emotion_sentiment', 
                'description_emotion_polarity', 'description_readability_score'
            ]
            available_cols = [col for col in display_cols if col in analyzed_df.columns]
            st.dataframe(analyzed_df[available_cols].head(20), use_container_width=True)

            # Download Results
            st.header("Download Analysis Results")
            csv = analyzed_df.to_csv(index=False)
            st.download_button(
                label="Download CSV with Analysis Results",
                data=csv,
                file_name="netflix_with_description_analysis.csv",
                mime="text/csv"
            )

    # Single Description Analysis
    st.header("Single Description Analysis")
    user_description = st.text_area(
        "Enter a Netflix-style description to analyze:",
        height=100,
        placeholder="Enter a movie or show description here..."
    )

    if user_description:
        # Analyze the user input
        polarity, subjectivity, sentiment = analyze_emotion(user_description)
        readability_score, readability_category = analyze_readability(user_description)

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Emotion Analysis")
            st.metric("Sentiment", sentiment.title())
            st.metric("Polarity", f"{polarity:.3f}")
            st.metric("Subjectivity", f"{subjectivity:.3f}")

        with col2:
            st.subheader("Readability Analysis")
            st.metric("Reading Level", readability_category.title())
            st.metric("Flesch Score", f"{readability_score:.1f}")

    else:
        st.info("Enter a description above to see live analysis results.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
        st.subheader("Error Details")
        st.code(traceback.format_exc())