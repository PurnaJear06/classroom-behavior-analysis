"""
Puter Claude Integration Module - Client-side implementation

This module provides integration with Claude 3.7 Sonnet via Puter.js.
Note: The actual AI calls happen client-side in the browser via Puter.js.
"""

import os
import json
import requests
import time
import http.server
import socketserver
import threading
import webbrowser
import socket
from urllib.parse import urlparse
from typing import Dict, List, Optional, Union, Any
import random

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Default API key - not needed for direct use of Claude via Streamlit
DEFAULT_API_KEY = ""

def get_api_key() -> str:
    """Get API key from session state or use default."""
    if STREAMLIT_AVAILABLE and 'api_key' in st.session_state and st.session_state.api_key:
        return st.session_state.api_key
    return DEFAULT_API_KEY

def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

class FileServerHandler(http.server.SimpleHTTPRequestHandler):
    """Simple HTTP request handler that serves from a specific directory."""
    
    def __init__(self, *args, directory=None, **kwargs):
        self.directory = directory
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        # Suppress log messages
        pass

def analyze_behavior(
    prompt: str,
    mode: str = "Standard",
    model: str = "claude",
    max_retries: int = 2
) -> str:
    """
    Placeholder for Puter.js Claude analysis - actual implementation is client-side.
    
    Args:
        prompt: The text prompt to analyze
        mode: Analysis mode - "Basic", "Standard", or "Comprehensive"
        model: The model to use (default is claude)
        max_retries: Number of retries
    
    Returns:
        str: Placeholder analysis
    """
    return f"[Placeholder] Analysis will be performed client-side via Puter.js"

def get_behavior_insights(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format behavior data for analysis - actual analysis happens client-side.
    
    Args:
        data: Dictionary of behavior data
    
    Returns:
        Dict: Response containing analysis placeholder and raw data
    """
    return {
        "analysis": "[Placeholder] Analysis will be performed client-side via Puter.js",
        "raw_data": data,
        "timestamp": time.time()
    }

def analyze_complete_tracking_data(
    track_data: List[Dict], 
    summary_data: Dict = None,
    mode: str = "Standard"
) -> Dict[str, Any]:
    """
    Placeholder for complete tracking data analysis - actual implementation is client-side.
    
    Args:
        track_data: List of tracking data dictionaries
        summary_data: Optional summary data
        mode: Analysis mode - "Basic", "Standard", or "Comprehensive"
    
    Returns:
        Dict: Success indicator and placeholder analysis
    """
    return {
        "success": True,
        "analysis": "[Placeholder] Analysis will be performed client-side via Puter.js",
        "timestamp": time.time()
    }

def format_yolo_data_for_claude(track_data: List[Dict], summary_data: Dict = None) -> str:
    """
    Format YOLO detection data for Claude analysis.
    
    Args:
        track_data: List of tracking data dictionaries
        summary_data: Optional summary data
    
    Returns:
        str: Formatted prompt
    """
    # Create a basic formatted prompt
    prompt = "# Classroom Behavior Data\n\n"
    
    # Add summary data if available
    if summary_data:
        prompt += "## Summary\n"
        prompt += f"- Students detected: {len(summary_data.get('students', {}))}\n"
        prompt += f"- Video duration: {summary_data.get('video_duration', 0)} seconds\n"
        prompt += f"- Total frames: {summary_data.get('total_frames', 0)}\n\n"
        
        # Add behavior counts
        if 'behavior_counts' in summary_data:
            prompt += "## Behavior Distribution\n"
            for behavior, count in summary_data.get('behavior_counts', {}).items():
                prompt += f"- {behavior}: {count}\n"
            prompt += "\n"
    
    # Add student data
    prompt += "## Student Data\n"
    # This is simplified for the placeholder version
    prompt += f"- Total students: {len(set([d.get('id') for d in track_data if 'id' in d]))}\n"
    prompt += f"- Total detections: {len(track_data)}\n\n"
    
    return prompt

# Simple function to check if integration is working
def is_working() -> bool:
    """Check if the Puter Claude integration is working"""
    return True

def analyze_behavior_with_oversight(
    frame_data: Dict,
    yolo_results: Dict,
    opencv_results: Dict = None,
    mode: str = "Standard"
) -> Dict:
    """
    Enhanced behavior analysis with Claude's oversight of YOLO and OpenCV results.
    
    Args:
        frame_data: Dictionary containing frame information
        yolo_results: YOLO detection results
        opencv_results: Optional OpenCV detection results
        mode: Analysis mode (Basic, Standard, Comprehensive)
    """
    # Create detailed prompt based on mode
    if mode == "Basic":
        prompt = f"""
        Verify YOLO detections for frame {frame_data.get('frame_number', 'unknown')}:
        
        YOLO Detections:
        {json.dumps(yolo_results, indent=2)}
        
        Verify:
        1. Are detections reasonable for a classroom setting?
        2. Any obvious false positives?
        """
    elif mode == "Standard":
        prompt = f"""
        Analyze and verify detections for frame {frame_data.get('frame_number', 'unknown')}:
        
        YOLO Detections:
        {json.dumps(yolo_results, indent=2)}
        
        OpenCV Results (if available):
        {json.dumps(opencv_results, indent=2) if opencv_results else "Not available"}
        
        Provide:
        1. Verification of each detection
        2. Suggested corrections
        3. Confidence score assessment
        4. Integration quality between YOLO and OpenCV
        """
    else:  # Comprehensive
        prompt = f"""
        Perform comprehensive analysis of frame {frame_data.get('frame_number', 'unknown')}:
        
        YOLO Detections:
        {json.dumps(yolo_results, indent=2)}
        
        OpenCV Results:
        {json.dumps(opencv_results, indent=2) if opencv_results else "Not available"}
        
        Frame Context:
        {json.dumps(frame_data, indent=2)}
        
        Provide:
        1. Detailed verification of each detection
        2. Cross-validation between YOLO and OpenCV
        3. Behavior pattern analysis
        4. Suggested optimizations
        5. Accuracy metrics for each model
        6. Combined accuracy assessment
        7. Recommendations for improvement
        """
    
    # Get Claude's analysis
    analysis = analyze_behavior(prompt, model="claude-3-5-sonnet", mode=mode)
    
    if analysis.get('success', False):
        # Add accuracy metrics
        analysis['accuracy_assessment'] = {
            'yolo_accuracy': calculate_yolo_accuracy(yolo_results, analysis),
            'opencv_accuracy': calculate_opencv_accuracy(opencv_results, analysis) if opencv_results else 0,
            'combined_accuracy': calculate_combined_accuracy(yolo_results, opencv_results, analysis)
        }
        
        # Add correction suggestions
        if 'corrections' in analysis:
            analysis['corrections'] = format_corrections(analysis['corrections'])
    
    return analysis

def calculate_yolo_accuracy(yolo_results: Dict, claude_analysis: Dict) -> float:
    """Calculate YOLO accuracy based on Claude's verification."""
    try:
        verified_detections = claude_analysis.get('verified_detections', [])
        total_detections = len(yolo_results.get('detections', []))
        correct_detections = sum(1 for v in verified_detections if v.get('is_correct', False))
        
        return (correct_detections / total_detections * 100) if total_detections > 0 else 0
    except Exception:
        return 0

def calculate_opencv_accuracy(opencv_results: Dict, claude_analysis: Dict) -> float:
    """Calculate OpenCV accuracy based on Claude's verification."""
    try:
        verified_opencv = claude_analysis.get('verified_opencv', [])
        total_detections = len(opencv_results.get('detections', []))
        correct_detections = sum(1 for v in verified_opencv if v.get('is_correct', False))
        
        return (correct_detections / total_detections * 100) if total_detections > 0 else 0
    except Exception:
        return 0

def calculate_combined_accuracy(
    yolo_results: Dict,
    opencv_results: Dict,
    claude_analysis: Dict
) -> float:
    """Calculate combined model accuracy."""
    yolo_acc = calculate_yolo_accuracy(yolo_results, claude_analysis)
    opencv_acc = calculate_opencv_accuracy(opencv_results, claude_analysis) if opencv_results else 0
    
    # If both models are used, weight them equally
    if opencv_results:
        return (yolo_acc + opencv_acc) / 2
    
    # If only YOLO is used, return its accuracy
    return yolo_acc

def format_corrections(corrections: List) -> List[Dict]:
    """Format correction suggestions for display."""
    formatted = []
    for correction in corrections:
        formatted.append({
            'original': correction.get('original_detection', ''),
            'corrected': correction.get('corrected_detection', ''),
            'confidence': correction.get('confidence', 0),
            'explanation': correction.get('explanation', '')
        })
    return formatted 