#!/usr/bin/env python3
"""
Test script to verify Claude integration functionality
"""

import sys
import os
import json
from unittest.mock import patch

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions to test
try:
    from claude_integration import (
        is_working,
        analyze_complete_tracking_data,
        format_yolo_data_for_claude
    )
    print("✅ Successfully imported all functions from claude_integration")
except ImportError as e:
    print(f"❌ Error importing functions: {e}")
    sys.exit(1)

def test_is_working():
    """Test if the is_working function returns True"""
    result = is_working()
    if result:
        print("✅ Integration is working")
        return True
    else:
        print("❌ Integration is not working")
        return False

def test_format_data():
    """Test formatting YOLO data for Claude analysis"""
    # Sample tracking data
    track_data = [
        {"frame": 1, "id": 1, "class": "Attentive", "confidence": 0.93},
        {"frame": 1, "id": 2, "class": "Writing Notes", "confidence": 0.87},
        {"frame": 2, "id": 1, "class": "Attentive", "confidence": 0.95}
    ]
    
    # Sample summary data
    summary_data = {
        "students": {1: {}, 2: {}},
        "video_duration": 120,
        "total_frames": 100,
        "behavior_counts": {"Attentive": 50, "Writing Notes": 30}
    }
    
    # Format data for Claude
    formatted_data = format_yolo_data_for_claude(track_data, summary_data)
    
    if (isinstance(formatted_data, str) and 
        "Classroom Behavior Data" in formatted_data and
        "Behavior Distribution" in formatted_data):
        print("✅ Successfully formatted data for Claude")
        return True
    else:
        print("❌ Failed to format data properly")
        return False

if __name__ == "__main__":
    print("=== Running Claude Integration Tests ===\n")
    
    # Run tests
    working = test_is_working()
    formatting = test_format_data()
    
    # Report results
    if all([working, formatting]):
        print("\n🎉 All tests passed! Claude integration is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the integration.")
        sys.exit(1) 