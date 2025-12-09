# Netflix Sentiment Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing Netflix content and audience sentiment using VADER and spaCy models. This project combines movie/TV show metadata with user reviews from TMDB (The Movie Database) to provide insights into content trends and viewer reactions.

# App Link:
https://introtodatasciencefinalproject-zvvaznzgerdexky9cmphbz.streamlit.app/

## 🎯 Features

- **Sentiment Analysis Pipeline**
  - VADER (rule-based) and spaCy (ML-based) sentiment analysis
  - Automated review fetching from TMDB API
  - Batch processing for enrichment and scoring

- **Interactive Dashboard**
  - 📊 Overview with key metrics and trends
  - 🔎 Detailed content exploration
  - ⚖️ Model comparison (VADER vs spaCy)
  - 🧭 Title-specific sentiment analysis
  - ⚙️ Data ingestion and scoring controls

- **Data Sources**
  - Netflix titles dataset (`netflix_titles.csv`)
  - TMDB API integration for reviews
  - Support for multiple review sources (TMDB, IMDB, Rotten Tomatoes)

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure TMDB API

- Get an API key from [TMDB](https://www.themoviedb.org/documentation/api)

- Create `.streamlit/secrets.toml` file:

```toml
[tmdb]
api_key = "YOUR_TMDB_V3_KEY"
```

- Or set environment variable: `TMDB_API_KEY`

### 3. Prepare Data

- Place your Netflix dataset in `data/netflix_titles.csv`
- Run the pipeline steps in the dashboard:
  1. Fetch TMDB Reviews
  2. Enrich & Score (VADER + spaCy)

## 📁 Project Structure

```text
├── app.py                 # Streamlit dashboard application
├── api_tmdb.py           # TMDB API client
├── api_clients.py        # Generic API utilities
├── cleaning.py           # Data cleaning utilities
├── nlp_utils.py          # NLP processing utilities
├── scraping.py          # Web scraping utilities
├── viz.py               # Visualization functions
├── data/                # Data storage
│   ├── netflix_titles.csv
│   ├── reviews_raw.csv
│   └── netflix_enriched_scored.csv
├── nlp/                 # NLP models and utilities
│   ├── spacy_model/
│   ├── vader_model/
│   └── utils/
└── scripts/             # Processing scripts
    ├── enrich_and_score.py
    ├── fetch_tmdb_reviews.py
    ├── run_inference.py
    └── train_spacy.py
```

## 🏃‍♂️ Running the Program

### Initial Setup

1. First, ensure all dependencies are installed:

   ```bash
   pip install -r requirements.txt
   ```

2. Set up your TMDB API key (one of two ways):

   ```bash
   # Option 1: Create .streamlit/secrets.toml
   mkdir .streamlit
   echo '[tmdb]`napi_key = "YOUR_TMDB_V3_KEY"' > .streamlit/secrets.toml

   # Option 2: Set environment variable
   $env:TMDB_API_KEY = "YOUR_TMDB_V3_KEY"  # PowerShell
   ```

### Data Preparation

1. Place the Netflix dataset in the data folder:

   ```bash
   mkdir -p data
   # Copy your netflix_titles.csv to the data folder
   ```

2. Launch the Streamlit dashboard:

   ```bash
   streamlit run app.py
   ```

### Using the Dashboard

1. **Pipeline Steps** (in the sidebar):
   - Click "🔄 Fetch TMDB Reviews" to collect review data
   - Click "⚙️ Enrich + Score" to process the reviews
   
2. **Navigate Tabs**:
   - 📊 **Overview**: General statistics and trends
   - 🔎 **Explore**: Detailed data exploration
   - ⚖️ **Model Compare**: VADER vs spaCy analysis
   - 🧭 **Title Explorer**: Individual title analysis
   - ⚙️ **Ingest & Score**: Data processing controls

### Command-line Scripts

You can also run the pipeline steps directly using scripts:

1. Fetch reviews:

   ```bash
   python scripts/fetch_tmdb_reviews.py --netflix "data/netflix_titles.csv" --output "data/reviews_raw.csv" --limit 300
   ```

2. Process and score reviews:

   ```bash
   python scripts/enrich_and_score.py --netflix "data/netflix_titles.csv" --reviews "data/reviews_raw.csv" --output "data/netflix_enriched_scored.csv" --spacyModel "nlp/spacy_model/artifacts/best"
   ```

3. Run model evaluation:

   ```bash
   python scripts/run_inference.py --input "data/test_reviews.csv" --output "data/predictions.csv" --spacyModel "nlp/spacy_model/artifacts/best"
   ```

### Troubleshooting

- If you see "Missing data/netflix_titles.csv" - Ensure the dataset is in the correct location
- If you get API errors - Verify your TMDB API key is set correctly
- For performance issues - Try reducing the `--limit` parameter when fetching reviews
- For memory issues - Process data in smaller batches

The dashboard will be available at [http://localhost:8501](http://localhost:8501)

## 📊 Pipeline Overview

1. **Data Collection**
   - Base Netflix catalog data
   - TMDB reviews fetching
   - Optional: IMDB/Rotten Tomatoes scraping

2. **Enrichment & Scoring**
   - VADER sentiment analysis
   - spaCy model inference
   - Metadata enrichment

3. **Analysis & Visualization**
   - Sentiment distribution
   - Content trends
   - Model comparison
   - Title-specific insights

## 🛠️ Tools & Technologies

- **Framework**: Streamlit
- **Data Processing**: Pandas
- **NLP**:
  - VADER (rule-based sentiment)
  - spaCy (custom trained model)
  - NLTK utilities
- **APIs**: TMDB
- **Visualization**: Built-in Streamlit charts

## 💡 Usage Tips

- Set your TMDB API key in `.streamlit/secrets.toml` or as an environment variable
- Start with a small dataset for testing
- Use the "Pipeline" section in the sidebar to process data
- Check the spaCy model evaluation on test data
- Explore different visualizations in each tab

## � Data Format Requirements

### Netflix Titles Dataset
Required columns in `netflix_titles.csv`:
- `show_id`: Unique identifier
- `type`: Content type (Movie/TV Show)
- `title`: Title of the content
- `release_year`: Year of release
- `description`: Content description

### Review Dataset
Generated `reviews_raw.csv` will contain:
- `show_id`: References netflix_titles.csv
- `type`: Content type
- `title`: Title
- `release_year`: Year
- `review_text`: The review content
- `author`: Reviewer name/ID
- `created_at`: Review timestamp
- `url`: Source URL
- `source`: Review source (e.g., "tmdb")

## 🎯 Model Training

### Training a New spaCy Model

```bash
python scripts/train_spacy.py \
  --train "data/train_reviews.csv" \
  --dev "data/dev_reviews.csv" \
  --test "data/test_reviews.csv" \
  --textCol "text" \
  --labelCol "label" \
  --nEpochs 10 \
  --lr 2e-3 \
  --dropout 0.2 \
  --batchSize 64 \
  --outputDir "nlp/spacy_model/artifacts"
```

Training data format:
- CSV file with at least two columns:
  - `text`: Review text
  - `label`: Sentiment ("positive"/"negative" or 1/0)

## 📦 Dependencies

Required Python packages:
```txt
pandas           # Data processing
numpy            # Numerical operations
plotly           # Visualization
streamlit        # Dashboard framework
requests         # API client
beautifulsoup4   # Web scraping
lxml             # XML/HTML parsing
nltk             # NLP utilities
tensorflow       # Deep learning
scikit-learn     # Machine learning
wordcloud        # Word cloud viz
pycountry        # Country code handling
spacy            # NLP modeling
tqdm             # Progress bars
vaderSentiment   # Rule-based sentiment
```

## 🔍 Advanced Usage

### Custom Review Collection

1. Using TMDB API directly:

   ```python
   from api_tmdb import fetchReviewsForTitle
   reviews = fetchReviewsForTitle(title="Stranger Things", year=2016, kind="tv")
   ```

1. Web scraping (IMDB/Rotten Tomatoes):

   ```python
   from scraping import scrapeReviews
   reviews = scrapeReviews(url="https://www.imdb.com/title/tt123456/reviews", site="imdb", maxPages=2)
   ```

### Batch Processing

For large datasets, use the command-line scripts with appropriate limits:

```bash
# Fetch reviews in batches
python scripts/fetch_tmdb_reviews.py --netflix "data/netflix_titles.csv" --limit 100

# Score a custom dataset
python scripts/run_inference.py --input "data/my_reviews.csv" --textCol "review_text"
```

## �📝 Notes

- TMDB API has rate limits - use the `--limit` parameter when fetching reviews
- Large datasets may take time to process
- The spaCy model can be retrained using custom data
- Web scraping functions respect site robots.txt and include appropriate delays
- NLTK data is downloaded automatically to `.nltk_data/`
- Model artifacts are saved to `nlp/spacy_model/artifacts/`
- Use CSV files with UTF-8 encoding for best compatibility

---

Created for INFO 501 Final Project (Fall 2025)

