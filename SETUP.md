# Environment Setup Guide

Choose one of the three methods below to set up your environment:

## Method 1: Using Conda (Recommended)

```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate netflixv1

# Run the app
streamlit run app.py
```

## Method 2: Using Python venv

```bash
# Create virtual environment
python3 -m venv netflixv1

# Activate environment (macOS/Linux)
source netflixv1/bin/activate

# Activate environment (Windows)
netflixv1\Scripts\activate

# For better Streamlit performance (macOS/Linux only)
xcode-select --install  # Install command line tools
pip install watchdog

# Install packages
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Method 3: Using pip (System-wide)

```bash
# For better Streamlit performance (macOS/Linux only)
xcode-select --install  # Install command line tools
pip install watchdog

# Install packages directly
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deactivate Environment

```bash
# For conda
conda deactivate

# For venv
deactivate
```

## Performance Optimization

For better Streamlit performance and file watching:

**macOS Users:**
```bash
# Install Xcode command line tools (if not already installed)
xcode-select --install

# Watchdog is included in requirements.txt for better file watching
```

**Note:** Watchdog is automatically installed with the requirements and provides better file watching for Streamlit's auto-reload feature.

## Test Environment

After setting up, test if everything works:

```bash
python test_env.py
```

## Remove Environment

```bash
# For conda
conda env remove -n netflixv1

# For venv
rm -rf netflixv1  # (macOS/Linux)
rmdir /s netflixv1  # (Windows)
```