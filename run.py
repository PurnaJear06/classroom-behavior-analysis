#!/usr/bin/env python3
"""
Classroom Behavior Analysis - Main Runner

This script starts the Streamlit app with the correct Python path.
"""

import os
import sys
import subprocess

def main():
    """Main function to run the Streamlit app"""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the current directory to the Python path
    sys.path.insert(0, current_dir)
    
    # Set model paths as environment variables
    os.environ['MODEL_PATH'] = os.path.join(current_dir, 'models', 'best.pt')
    os.environ['FACE_MODEL_PATH'] = os.path.join(current_dir, 'models', 'res10_300x300_ssd_iter_140000.caffemodel')
    os.environ['FACE_PROTOTXT_PATH'] = os.path.join(current_dir, 'models', 'deploy.prototxt')
    
    print("Starting Classroom Behavior Analysis App...")
    print("Models directory:", os.path.join(current_dir, 'models'))
    
    # Run the Streamlit app
    subprocess.call(['streamlit', 'run', os.path.join(current_dir, 'src', 'app.py')])

if __name__ == "__main__":
    main() 