#!/usr/bin/env python3
"""
Test script to verify Puter.js integration functionality
"""

import sys
import os
import json
from unittest.mock import patch

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions to test
try:
    from puter_integration import (
        is_working,
        analyze_behavior_with_puter,
        get_behavior_insights
    )
    print("✅ Successfully imported all functions from puter_integration")
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

def test_analysis():
    """Test the behavior analysis function"""
    prompt = "Analyze classroom behavior with students showing various engagement levels."
    
    # Test with different modes
    modes = ["Basic", "Standard", "Comprehensive"]
    for mode in modes:
        print(f"Testing {mode} analysis mode...")
        result = analyze_behavior_with_puter(prompt, mode=mode)
        
        if result and isinstance(result, str) and len(result) > 50:
            print(f"✅ Successfully received {mode} analysis response")
        else:
            print(f"❌ Failed to get proper {mode} analysis response")
            return False
    
    return True

if __name__ == "__main__":
    print("=== Running Puter.js Integration Tests ===\n")
    
    # Run tests
    working = test_is_working()
    analysis_success = test_analysis()
    
    # Report results
    if all([working, analysis_success]):
        print("\n🎉 All tests passed! Puter.js integration is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the integration.")
        sys.exit(1) 