@echo off
echo 🎬 Netflix Original vs Licensed Classifier
echo ==========================================
echo.
echo Starting intelligent Netflix content analysis...
echo This app identifies Netflix Originals vs Licensed Content
echo.

cd /d "%~dp0"

REM Activate the virtual environment
call reports\environment\netflix_env\Scripts\activate.bat

REM Check if activation worked
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Run the Netflix Original vs Licensed Classifier app
echo Starting Netflix Original vs Licensed Classifier...
python -m streamlit run src/analysis/netflix_original_vs_licensed_classifier_app.py

REM Deactivate the virtual environment
deactivate

pause