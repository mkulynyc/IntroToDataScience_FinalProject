@echo off
echo 📊 Netflix Description Analyzer - Streamlit Web App
echo ===================================================
echo.
echo Starting Streamlit web application...
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

REM Run the Streamlit app
echo Starting Netflix Description Analyzer...
python -m streamlit run src/analysis/netflix_description_analyzer_app.py

REM Deactivate the virtual environment
deactivate

pause