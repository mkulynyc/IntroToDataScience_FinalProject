# Netflix Movie Recommender System

This project contains a comprehensive data science pipeline for analyzing Netflix content and building recommendation systems using various mathematical and statistical techniques.

## Overview

The code has been converted from Jupyter notebook (`mathDS.ipynb`) into a well-structured Python module (`recomender_v1.py`) with the following main components:

### Classes and Functionality

1. **NetflixDataProcessor**: Data loading, cleaning, and preprocessing
2. **StatisticalAnalyzer**: Descriptive statistics and probability distributions
3. **BayesianInference**: Bayesian classification using Naive Bayes
4. **LinearAlgebraAnalyzer**: Matrix operations, PCA, and dimensionality reduction
5. **RegressionAnalyzer**: Linear and logistic regression models
6. **ClusteringAnalyzer**: K-means clustering and similarity analysis
7. **TimeSeriesAnalyzer**: Trend analysis and forecasting
8. **RecommendationEngine**: Content-based recommendation system
9. **NetflixAnalyticsPipeline**: Main orchestration class

## Installation & Environment Setup

Choose one of the following methods to set up your environment:

### Method 1: Using Conda (Recommended)

```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate netflixv1

# Run the app
streamlit run app.py
```

### Method 2: Using Python venv

```bash
# Create virtual environment
python3 -m venv netflixv1

# Activate environment (macOS/Linux)
source netflixv1/bin/activate

# Activate environment (Windows)
netflixv1\Scripts\activate

# Install packages
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Method 3: Using pip (System-wide)

```bash
# Install packages directly
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

📋 See `SETUP.md` for detailed setup instructions.

## Usage

### Quick Start

```python
from recomender_v1 import NetflixAnalyticsPipeline

# Initialize the pipeline
pipeline = NetflixAnalyticsPipeline()

# Run the complete analysis (update paths to your data files)
results = pipeline.run_full_pipeline(
    netflix_path="netflix_titles.csv",
    runtime_path="netflix_movies_tv_runtime.csv"  # Optional
)

# Create visualizations
pipeline.create_visualizations()

# Get recommendations
recommendations = pipeline.get_recommendations("Stranger Things", method='genre', top_n=5)
print(recommendations)
```

### Individual Component Usage

```python
from recomender_v1 import NetflixDataProcessor, StatisticalAnalyzer, RecommendationEngine

# Data processing only
processor = NetflixDataProcessor()
df = processor.load_data("netflix_titles.csv")
df = processor.clean_data()
df = processor.normalize_duration()

# Statistical analysis
stats = StatisticalAnalyzer(df)
stats.descriptive_statistics()
stats.plot_distributions()

# Recommendation system
recommender = RecommendationEngine(df)
recommendations = recommender.get_recommendations_by_genre("Movie Title", top_n=10)
```

## Features

### Data Processing
- Comprehensive data cleaning and normalization
- Handling of multi-value fields (genres, countries, cast, directors)
- Duration parsing for movies and TV shows
- Duplicate removal and data validation

### Statistical Analysis
- Descriptive statistics for numerical variables
- Distribution analysis and visualization
- Interactive plots using Plotly
- Probability density estimation

### Machine Learning Models
- **Bayesian Inference**: Naive Bayes classifier for content type prediction
- **Linear Regression**: Movie duration prediction
- **Logistic Regression**: Binary classification (Movie vs TV Show)
- **K-Means Clustering**: Content grouping based on features
- **PCA**: Dimensionality reduction for genre analysis

### Recommendation System
- Genre-based similarity using cosine similarity
- Description-based similarity using sentence embeddings
- Interactive recommendation interface
- Support for both collaborative and content-based filtering

### Time Series Analysis
- Content addition trends over time
- Moving averages and trend analysis
- Exponential smoothing for forecasting
- Interactive time series visualizations

### Advanced Analytics
- Hierarchical clustering of countries
- Feature importance analysis
- Distance metrics (Cosine, Jaccard)
- Network analysis capabilities

## Data Requirements

The system expects the following CSV files:

1. **netflix_titles.csv** (Required): Main Netflix dataset with columns:
   - show_id, type, title, director, cast, country, date_added
   - release_year, rating, duration, listed_in, description

2. **netflix_movies_tv_runtime.csv** (Optional): Additional runtime information

## Key Mathematical Concepts Implemented

- **Probability & Statistics**: Descriptive statistics, probability distributions, Bayesian inference
- **Linear Algebra**: Matrix operations, PCA, vectorization, similarity metrics
- **Calculus & Optimization**: Gradient descent concepts, loss functions
- **Clustering & Distance Metrics**: K-means, hierarchical clustering, cosine similarity
- **Time Series Analysis**: Trend analysis, forecasting, moving averages

## Visualizations

The system generates various interactive visualizations:
- Distribution plots (histograms with KDE)
- PCA scatter plots color-coded by different attributes
- Time series plots with trend lines and forecasts
- Clustering visualizations
- Heatmaps for similarity matrices
- Hierarchical clustering dendrograms

## Example Outputs

### Sample Recommendation Output
```
Recommendations for 'Stranger Things':
                    Title      Type Rating  Release Year  Similarity Score
0              Dark         TV Show  TV-MA          2017              0.89
1         The Umbrella Academy  TV Show  TV-14          2019              0.85
2              Manifest         TV Show  TV-14          2018              0.82
3         Black Mirror         TV Show  TV-MA          2011              0.78
4         The Haunting of Hill House TV Show TV-MA 2018              0.75
```

## Notes

- Make sure to update file paths in the main() function to match your data location
- Some features require additional libraries (sentence-transformers for semantic similarity)
- The system handles missing data gracefully with appropriate imputation strategies
- All visualizations are interactive using Plotly for better exploration

## Future Enhancements

- Integration with real-time data sources
- User-based collaborative filtering
- Deep learning models for improved recommendations
- A/B testing framework for recommendation evaluation
- Web interface using Streamlit or Flask

## Streamlit Web Application

A comprehensive web interface is available for interactive exploration:

### Quick Start with Streamlit

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Launch the app**:
```bash
python run_app.py
# OR
streamlit run app.py
```

3. **Open your browser** to http://localhost:8501

### Streamlit Features

- 📊 **Interactive Dashboard**: Real-time data exploration
- 🎯 **Recommendation Engine**: Get personalized content suggestions
- 📈 **Statistical Analysis**: Interactive charts and graphs
- 🔍 **Advanced Analytics**: Machine learning model results
- ⏰ **Time Series**: Trend analysis and forecasting
- 🎨 **Visualizations**: Plotly-powered interactive charts
- 💾 **Pickle Integration**: Fast loading with cached data

### App Components

1. **Overview Tab**: Dataset statistics and basic visualizations
2. **Recommendations Tab**: Interactive recommendation engine
3. **Statistics Tab**: Detailed statistical analysis
4. **Analysis Tab**: Machine learning model results
5. **Time Series Tab**: Temporal analysis and trends
6. **Visualizations Tab**: Advanced interactive charts

### Environment Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Pip dependencies |
| `environment.yml` | Conda environment |
| `SETUP.md` | Detailed setup guide |

### Performance Optimization

The app uses pickle files for fast loading:
- First run: Processes data and saves to pickle files (may take 5-10 minutes)
- Subsequent runs: Loads from pickle files (loads in seconds)
- Use "Force Reload" to reprocess data if needed

### Data Processing Scripts

- **`data_processor.py`**: Standalone script to process and pickle data
- **`run_app.py`**: Smart launcher that checks requirements and data

```bash
# Process data separately (optional)
python data_processor.py

# Check pickle files
python data_processor.py check
```

## Contributing

Feel free to contribute by:
- Adding new recommendation algorithms
- Improving data preprocessing
- Adding more visualization options
- Enhancing the evaluation metrics
- Improving the Streamlit interface
- Adding new interactive features