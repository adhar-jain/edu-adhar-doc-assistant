#!/usr/bin/env python3
"""
run_app.py - Application launcher

This script properly sets up the Python path and launches the Streamlit app.
Usage: python run_app.py
"""
import subprocess
import sys
import os
from pathlib import Path

# Set project root as current directory
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)

# Run Streamlit app
subprocess.run(
    [sys.executable, "-m", "streamlit", "run", "app/main.py"],
    env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
)
