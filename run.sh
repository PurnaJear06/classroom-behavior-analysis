#!/bin/bash

# Classroom Behavior Analysis Runner Script

# Navigate to the project directory (where this script is located)
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Creating new environment..."
    python -m venv venv
    source venv/bin/activate
    
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the application
echo "Starting Classroom Behavior Analysis Application..."
python run.py

# Keep terminal open if error occurs
read -p "Press Enter to exit..." 