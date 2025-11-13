@echo off
REM Enhanced Netflix Description Analyzer Launcher
REM Created for easy access to the advanced NLP analysis features
REM This version includes AI summaries and keyword extraction

echo.
echo 🎭 Enhanced Netflix Description Analyzer
echo ========================================
echo.
echo Loading enhanced dataset with pre-computed NLP features...
echo Dataset: netflix_with_enhanced_description_analysis.csv
echo.
echo Features included:
echo   - AI-powered content summaries
echo   - Sentiment and emotion analysis  
echo   - Readability scoring
echo   - Keyword extraction
echo.
echo Starting web interface...

cd /d "%~dp0"

REM Activate the virtual environment
call reports\environment\netflix_env\Scripts\activate.bat

REM Check if activation worked
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Run the Enhanced Streamlit app
echo Starting Enhanced NLP Description Analyzer...
python -m streamlit run src/analysis/enhanced_description_analyzer/enhanced_netflix_description_analyzer_app.py

REM Deactivate the virtual environment
deactivate

pause