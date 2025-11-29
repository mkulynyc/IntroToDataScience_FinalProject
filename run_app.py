#!/usr/bin/env python3
"""
11/29/2025
Netflix Recommender System Launcher
Easy startup script for the Streamlit application
"""

import os
import sys
import subprocess
import socket

DEFAULT_PORT = 8501
PORT_ENV_VAR = "STREAMLIT_PORT"
PORT_SEARCH_RANGE = range(DEFAULT_PORT, DEFAULT_PORT + 10)

def find_open_port():
    """
    Determine a usable TCP port for Streamlit.
    Respects STREAMLIT_PORT env override and falls back to scanning a range.
    """
    # Helper to test port availability
    def port_is_open(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("localhost", port))
                return True
            except OSError:
                return False

    # Respect explicit override first
    if PORT_ENV_VAR in os.environ:
        try:
            override = int(os.environ[PORT_ENV_VAR])
            if port_is_open(override):
                return override
            print(f"⚠️ Requested port {override} (via {PORT_ENV_VAR}) is unavailable. Falling back to auto-selection.")
        except ValueError:
            print(f"⚠️ Invalid {PORT_ENV_VAR} value: {os.environ[PORT_ENV_VAR]!r}. Falling back to auto-selection.")

    # Scan the default range
    for port in PORT_SEARCH_RANGE:
        if port_is_open(port):
            return port

    raise RuntimeError("Unable to find an open TCP port for Streamlit.")

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import sklearn
        print("✅ All required packages are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {str(e)}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_data_files():
    """Check if data files are available"""
    netflix_file = "netflix_titles.csv"
    pickle_dir = "pickle_data"
    
    has_csv = os.path.exists(netflix_file)
    has_pickle = (os.path.exists(os.path.join(pickle_dir, "netflix_pipeline.pkl")) and 
                  os.path.exists(os.path.join(pickle_dir, "netflix_data.pkl")))
    
    if has_pickle:
        print("✅ Pickle files found - app will load quickly!")
        return True
    elif has_csv:
        print("📁 CSV file found - will process on first run")
        return True
    else:
        print("❌ No data files found!")
        print("Please ensure 'netflix_titles.csv' is in the current directory")
        return False

def run_streamlit():
    """Launch the Streamlit application"""
    try:
        port = find_open_port()
        address = "localhost"
        print("🚀 Starting Netflix Recommender System...")
        print(f"🌐 Opening browser at http://{address}:{port}")
        cmd = [
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            str(port),
            "--server.address",
            address,
        ]
        subprocess.run(cmd, check=True)
    except RuntimeError as err:
        print(f"❌ {err}")
        print("Try setting STREAMLIT_PORT to a port you can bind to (e.g. 8505).")
    except subprocess.CalledProcessError as err:
        print("❌ Error running Streamlit")
        if err.returncode == 1:
            print("Streamlit reported a permissions or bind issue.")
        print("Make sure Streamlit is installed and accessible: pip install streamlit")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

def main():
    """Main launcher function"""
    print("Netflix Recommender System Launcher")
    print("===================================")
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check data
    if not check_data_files():
        return
    
    # Run Streamlit
    run_streamlit()

if __name__ == "__main__":
    main()
