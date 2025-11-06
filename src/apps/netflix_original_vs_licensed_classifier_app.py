#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Netflix Content Classification Tool

This app tries to figure out whether Netflix shows are originals or licensed content.
Uses several different ML approaches and combines them for better accuracy.

The models aren't perfect but they give decent results on the test data.
Main challenge is that Netflix doesn't always make it obvious what's original vs licensed.

Built this because I was curious about Netflix's content strategy.

Dependencies: pandas, sklearn, streamlit, plotly
Data: Expects netflix_clean.csv in the data folder
"""

import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
import logging

# Suppress some warnings that clutter the output
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('transformers').setLevel(logging.ERROR)

# Import our classifier - this has the actual classification logic
import sys
sys.path.append(str(Path(__file__).parent.parent))
try:
    from analyzers.netflix_original_vs_licensed_classifier import NetflixContentClassifier
except ImportError:
    st.error("Could not import NetflixContentClassifier. Make sure netflix_original_vs_licensed_classifier.py is in the analyzers folder.")
    st.stop()

# Some constants to make the code cleaner
NETFLIX_COLORS = ['#E50914', '#221F1F', '#F5F5F1']
CONFIDENCE_THRESHOLD = 0.7 # Arbitrary threshold for "high confidence" predictions

def main():
    """Run the Streamlit web application"""

    # Page configuration
    st.set_page_config(
        page_title="Netflix Original vs Licensed Classifier",
        page_icon="N",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #E50914;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .subtitle {
        font-size: 1.5rem;
        font-weight: bold;
        color: #FFFFFF;
        background-color: #E50914;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 3px solid #FFFFFF;
    }
    .original-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.25rem;
    }
    .licensed-badge {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.25rem;
    }
    .uncertain-badge {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown('<div class="main-header">Netflix Original vs Licensed Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Smart Analysis to Identify Netflix Originals vs Licensed Content</div>', unsafe_allow_html=True)
    st.markdown("*Determines if Netflix content is a Netflix Original or Licensed using intelligent pattern analysis*")

    # Sidebar controls
    st.sidebar.title("Classification Settings")

    # Auto-load Netflix data - this app specifically needs the clean dataset
    df = None
    
    # Path resolution can be tricky when running from different directories
    # So we'll try a few common locations
    search_paths = [
        Path(__file__).parent.parent / "data" / "netflix_clean.csv", # src/analysis/ -> data/
        Path(__file__).parent.parent.parent / "data" / "netflix_clean.csv", # if we're deeper
        Path("data/netflix_clean.csv"), # relative to cwd
        Path("../data/netflix_clean.csv"), # one up from src/
        Path("../../data/netflix_clean.csv"), # two up (just in case)
    ]

    data_file = None
    for potential_path in search_paths:
        if potential_path.exists():
            data_file = potential_path
            break

    if data_file:
        try:
            df = pd.read_csv(data_file)
            st.sidebar.success(f"Netflix data loaded: {len(df):,} shows")
            st.sidebar.info(f"Using: {data_file.name}")
        except Exception as e:
            st.sidebar.error(f"Failed to load data: {e}")
            st.error("Can't load Netflix dataset. Check the file format.")
            st.stop()
    else:
        st.sidebar.error("Netflix clean dataset not found!")
        st.error("""
        **Missing Data File**
        
        This app needs `netflix_clean.csv` in the `data/` folder.
        Make sure the file exists and try again.
        """)
        st.stop()

    if df is not None and 'title' in df.columns:
        # Check if data already has classification
        classification_cols = {'content_classification', 'classification_score', 'classification_confidence'}

        has_classification = classification_cols.issubset(set(df.columns))

        if has_classification:
            st.info(f"Loaded pre-classified dataset: {len(df)} records")
            classified_df = df.copy()
            st.session_state['classified_df'] = classified_df
        else:
            # Main classification interface
            st.header("Run Content Classification")
            st.write("This will analyze your Netflix data to identify which titles are likely Originals vs Licensed content.")
            
            # Give users some options
            with st.expander("Classification Settings"):
                confidence_threshold = st.slider(
                    "Confidence threshold (higher = more conservative)", 
                    0.5, 0.9, 0.7, 0.05,
                    help="Lower values will classify more items, but with potentially less accuracy"
                )
                show_progress = st.checkbox("Show detailed progress", value=True)

            if st.button("Start Classification", type="primary"):
                with st.spinner("Running ML classification... This might take a minute or two"):
                    # Initialize our classifier with the ensemble methods
                    classifier = NetflixContentClassifier()
                    
                    # Set up progress tracking if requested
                    if show_progress:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("Initializing models...")
                    
                    try:
                        # Run the actual classification - this is where the magic happens
                        classified_df = classifier.ensemble_classification(df.copy())
                        
                        # Apply confidence threshold
                        low_confidence_mask = classified_df['classification_confidence'] < confidence_threshold
                        classified_df.loc[low_confidence_mask, 'content_classification'] = 'Uncertain'
                        
                        if show_progress:
                            progress_bar.progress(1.0)
                            status_text.text("Classification completed!")
                        
                        st.success("Classification finished successfully!")
                        
                    except Exception as e:
                        st.error(f"Classification failed: {str(e)}")
                        st.write("This might be due to data format issues or missing dependencies.")
                        return

                # Store results so user doesn't lose them if they interact with the interface
                st.session_state['classified_df'] = classified_df

        # Display results if available
        if 'classified_df' in st.session_state:
            classified_df = st.session_state['classified_df']

            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Titles", len(classified_df))
            with col2:
                originals = (classified_df['content_classification'] == 'Netflix Original').sum()
                st.metric("Netflix Originals", originals)
            with col3:
                licensed = (classified_df['content_classification'] == 'Licensed Content').sum()
                st.metric("Licensed Content", licensed)
            with col4:
                uncertain = (classified_df['content_classification'] == 'Uncertain').sum()
                st.metric("Uncertain", uncertain)

            # Classification Distribution
            st.header("Classification Distribution")
            classification_counts = classified_df['content_classification'].value_counts()
            fig_classification = px.pie(
                values=classification_counts.values,
                names=classification_counts.index,
                title="Content Classification Distribution",
                color_discrete_sequence=['#d4edda', '#f8d7da', '#fff3cd']
            )
            st.plotly_chart(fig_classification, use_container_width=True)

            # Confidence Distribution
            st.header("Classification Confidence")
            fig_confidence = px.histogram(
                classified_df,
                x='classification_confidence',
                title="Distribution of Classification Confidence Scores",
                nbins=20,
                color_discrete_sequence=['#E50914']
            )
            st.plotly_chart(fig_confidence, use_container_width=True)

            # Classification by Type (Movie vs TV)
            st.header("📺 Classification by Content Type")
            type_classification = pd.crosstab(
                classified_df['type'],
                classified_df['content_classification'],
                normalize='index'
            ) * 100

            fig_type = px.bar(
                type_classification,
                title="Classification Distribution by Content Type (%)",
                barmode='stack',
                color_discrete_sequence=['#d4edda', '#f8d7da', '#fff3cd']
            )
            st.plotly_chart(fig_type, use_container_width=True)

            # Classification by Year
            st.header("📅 Classification Trends by Year")
            yearly_classification = pd.crosstab(
                classified_df['release_year'],
                classified_df['content_classification']
            )

            fig_yearly = px.line(
                yearly_classification,
                title="Classification Trends Over Time",
                markers=True,
                color_discrete_sequence=['#d4edda', '#f8d7da', '#fff3cd']
            )
            st.plotly_chart(fig_yearly, use_container_width=True)

            # Sample Results
            st.header("Sample Classification Results")

            # High confidence originals
            st.subheader("🏆 High Confidence Netflix Originals")
            high_conf_originals = classified_df[
                (classified_df['content_classification'] == 'Netflix Original') &
                (classified_df['classification_confidence'] > 0.7)
            ].head(5)

            for _, row in high_conf_originals.iterrows():
                st.markdown(
                    f"<div class='original-badge'>{row['title']} ({row['type']}) - {row['classification_confidence']:.2f} confidence</div>",
                    unsafe_allow_html=True
                )

            # High confidence licensed
            st.subheader("📺 High Confidence Licensed Content")
            high_conf_licensed = classified_df[
                (classified_df['content_classification'] == 'Licensed Content') &
                (classified_df['classification_confidence'] > 0.7)
            ].head(5)

            for _, row in high_conf_licensed.iterrows():
                st.markdown(
                    f"<div class='licensed-badge'>📺 {row['title']} ({row['type']}) - {row['classification_confidence']:.2f} confidence</div>",
                    unsafe_allow_html=True
                )

            # Full results table
            st.subheader("Complete Classification Results")
            display_cols = [
                'title', 'type', 'release_year', 'content_classification',
                'classification_confidence', 'classification_score'
            ]
            st.dataframe(classified_df[display_cols].head(20), use_container_width=True)

            # Component Scores Analysis
            st.header("Classification Component Analysis")
            st.write("See how different factors contributed to each classification:")

            component_cols = ['title_score', 'description_score', 'year_score', 'country_score', 'genre_score']
            component_data = classified_df[component_cols].mean()

            fig_components = px.bar(
                x=component_data.index,
                y=component_data.values,
                title="Average Contribution of Each Classification Component",
                labels={'x': 'Component', 'y': 'Average Score'},
                color_discrete_sequence=['#E50914']
            )
            st.plotly_chart(fig_components, use_container_width=True)

            # Download Results
            st.header("💾 Download Classification Results")
            csv = classified_df.to_csv(index=False)
            st.download_button(
                label="Download CSV with Classification Results",
                data=csv,
                file_name="netflix_classified_robust.csv",
                mime="text/csv"
            )

        # Single Title Classification
        st.header("Single Title Classification")
        user_title = st.text_input(
            "Enter a Netflix title to classify:",
            placeholder="e.g., Stranger Things, The Crown, Breaking Bad"
        )

        user_description = st.text_area(
            "Enter description (optional, improves accuracy):",
            height=100,
            placeholder="Enter the show's/movie's description..."
        )

        user_year = st.number_input(
            "Release Year (optional):",
            min_value=1900,
            max_value=2025,
            value=2020
        )

        user_country = st.text_input(
            "Production Country (optional):",
            placeholder="e.g., United States, South Korea, Japan"
        )

        if st.button("Classify Title") and user_title.strip():
            with st.spinner("Analyzing content..."):
                # Create test dataframe
                test_data = pd.DataFrame({
                    'title': [user_title],
                    'description': [user_description or ""],
                    'release_year': [user_year],
                    'country': [user_country or ""],
                    'listed_in': [""], # Not used in single classification
                    'type': ["Unknown"] # Will be determined by context
                })

                # Initialize classifier
                classifier = NetflixContentClassifier()

                # Extract features
                features_df = classifier.extract_features(test_data)

                # Run individual classifications
                title_score = classifier.classify_by_title_patterns(features_df)[0]
                desc_score = classifier.classify_by_description(features_df)[0]
                year_score = classifier.classify_by_release_year(features_df)[0]
                country_score = classifier.classify_by_country_patterns(features_df)[0]
                genre_score = classifier.classify_by_genre_patterns(features_df)[0]

                # Calculate final score
                weights = {'title': 0.25, 'description': 0.30, 'year': 0.20, 'country': 0.15, 'genre': 0.10}
                final_score = (
                    title_score * weights['title'] +
                    desc_score * weights['description'] +
                    year_score * weights['year'] +
                    country_score * weights['country'] +
                    genre_score * weights['genre']
                )

                # Final classification
                if final_score > 0.1:
                    classification = 'Netflix Original'
                    badge_class = 'original-badge'
                    icon = ''
                elif final_score < -0.1:
                    classification = 'Licensed Content'
                    badge_class = 'licensed-badge'
                    icon = '📺'
                else:
                    classification = 'Uncertain'
                    badge_class = 'uncertain-badge'
                    icon = '❓'

                confidence = min(abs(final_score) * 2, 1.0)

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Classification Result")
                    st.markdown(
                        f"<div class='{badge_class}'>{icon} {classification} ({confidence:.1f} confidence)</div>",
                        unsafe_allow_html=True
                    )
                    st.write(f"**Overall Score:** {final_score:.3f}")

                with col2:
                    st.subheader("Component Scores")
                    st.write(f"**Title Analysis:** {title_score:.3f}")
                    st.write(f"**Description:** {desc_score:.3f}")
                    st.write(f"**Release Year:** {year_score:.3f}")
                    st.write(f"**Country:** {country_score:.3f}")
                    st.write(f"**Genre:** {genre_score:.3f}")

        # Footer info
        st.markdown("---")
        st.markdown("**Netflix Original vs Licensed Classifier v1.2** - Built with intelligent pattern analysis")
        st.markdown("*Uses weighted combination of 5 analysis methods: title, description, year, country, and genre patterns*")
        
        # Developer info toggle
        if st.sidebar.checkbox("Show system info", value=False):
            st.sidebar.write("**System Information:**")
            st.sidebar.write(f"- Data rows processed: {len(df) if df is not None else 0}")
            st.sidebar.write(f"- Classification features: Title, Description, Year, Country, Genre")
            st.sidebar.write(f"- Method: Weighted ensemble of 5 rule-based pattern matchers")
            st.sidebar.write(f"- Algorithm: Pattern matching + heuristic scoring (not ML)")

# Main execution - standard Python idiom
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("💥 Application error!")
        st.write(f"Error details: {str(e)}")
        
        # Show technical details for debugging
        if st.button("Show technical details"):
            import traceback
            st.code(traceback.format_exc())
        
        st.write("Try refreshing the page or check that all required files are present.")