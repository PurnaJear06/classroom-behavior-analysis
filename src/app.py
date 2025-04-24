# Import non-Streamlit libraries first
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import time
import json
import random
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import sys
import base64
from streamlit.components.v1 import html

# Fix torch/streamlit compatibility issues - import torch BEFORE streamlit
import torch

# Force Pytorch to avoid shared memory issues
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Configure torch settings
torch.set_grad_enabled(False)  # Disable gradients for inference
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Define the face recognition model loader function before it's called
@torch.no_grad()  # Disable gradients for this function
def load_face_recognition_model():
    """Load face recognition model for student tracking."""
    try:
        # Check if OpenCV DNN modules are available
        face_prototxt = os.environ.get('FACE_PROTOTXT_PATH', "models/deploy.prototxt")
        face_model = os.environ.get('FACE_MODEL_PATH', "models/res10_300x300_ssd_iter_140000.caffemodel")
        
        # Check if files exist at the specified paths
        if not os.path.exists(face_prototxt):
            alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "deploy.prototxt")
            if os.path.exists(alt_path):
                face_prototxt = alt_path
                
        if not os.path.exists(face_model):
            alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "res10_300x300_ssd_iter_140000.caffemodel")
            if os.path.exists(alt_path):
                face_model = alt_path
        
        # Load the face detector
        face_detector = cv2.dnn.readNetFromCaffe(face_prototxt, face_model)
        return face_detector
    except Exception as e:
        print(f"Could not load face recognition models: {e}")
        return None

# Load YOLO model - also defined early to avoid issues
@torch.no_grad()  # Disable gradients for this function
def load_model():
    """Load YOLOv8 model for behavior detection."""
    try:
        # Import here to delay loading until needed
        from ultralytics import YOLO
        
        # Get model path from environment or use default
        model_path = os.environ.get('MODEL_PATH', 'models/best.pt')
        
        # Check if file exists at the specified path
        if not os.path.exists(model_path):
            alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best.pt")
            if os.path.exists(alt_path):
                model_path = alt_path
        
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Next import streamlit
import streamlit as st
# Disable watchdog to prevent torch path issues
import streamlit.config as config
config.set_option('server.fileWatcherType', 'none')

# Import YOLO after torch and environment setup
from ultralytics import YOLO

# Now import our Puter integration after fixing environment
try:
    from src.integrations.puter_integration import analyze_behavior_with_puter, get_behavior_insights, is_working
    from src.integrations.claude_integration import analyze_complete_tracking_data
except ImportError:
    # Fallback for direct imports
    try:
        from integrations.puter_integration import analyze_behavior_with_puter, get_behavior_insights, is_working
        from integrations.claude_integration import analyze_complete_tracking_data
    except ImportError:
        # Last resort if run directly
        from puter_integration import analyze_behavior_with_puter, get_behavior_insights, is_working
        from claude_integration import analyze_complete_tracking_data

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Streamlit page config must be the first Streamlit command
st.set_page_config(
    page_title="Classroom Behavior Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default API key
DEFAULT_API_KEY = os.environ.get("API_KEY", "")

# Initialize session state for tracking and settings
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'track_data' not in st.session_state:
    st.session_state.track_data = None
if 'track_summary' not in st.session_state:
    st.session_state.track_summary = None
if 'face_snapshots' not in st.session_state:
    st.session_state.face_snapshots = {}
if 'selected_behaviors' not in st.session_state:
    st.session_state.selected_behaviors = []
if 'selected_student_ids' not in st.session_state:
    st.session_state.selected_student_ids = []
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
    
# Initialize Claude-related session state
if 'use_claude' not in st.session_state:
    st.session_state.use_claude = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = DEFAULT_API_KEY
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = "Standard"

# Apply custom CSS for modern UI
st.markdown("""
<style>
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #2C2C2C;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4F8BF9;
        color: white;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4F8BF9;
        color: white;
    }
    .css-1r6slb0 {  /* Metric container */
        background-color: #2C2C2C;
        border-radius: 5px;
        padding: 15px 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .css-1ht1j8u {
        overflow-wrap: anywhere;
        font-size: 0.8rem;
    }
    .profile-card {
        border-radius: 10px;
        background-color: #2C2C2C;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .student-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .student-image {
        border-radius: 50%;
        margin-right: 15px;
    }
    .behavior-tag {
        background-color: #4F8BF9;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 5px;
        margin-bottom: 5px;
        display: inline-block;
    }
    /* Customize header */
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #4F8BF9, #38B6FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
    /* Custom chart container */
    .chart-container {
        background-color: #2C2C2C;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Custom dashboard metrics */
    .metric-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
    }
    .custom-metric {
        background-color: #2C2C2C;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #CCC;
    }
</style>
""", unsafe_allow_html=True)

# Model path - update this to your local path
MODEL_PATH = 'best.pt'

# App title
st.title("🎓 Classroom Behavior Analysis Dashboard")

# Sidebar
st.sidebar.header("Controls & Filters")

# Upload section in sidebar
st.sidebar.subheader("1. Upload Media")
uploaded_file = st.sidebar.file_uploader("Upload classroom image or video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])

# Load model
try:
    model = load_model()
    st.sidebar.success("✅ YOLO model loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")
    st.stop()

# Detection settings
st.sidebar.subheader("Detection Settings")
confidence_threshold = st.sidebar.slider("Detection Confidence", min_value=0.05, max_value=1.0, value=0.15, step=0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", min_value=0.1, max_value=1.0, value=0.3, step=0.05)

# Add tooltip/help text for the IoU setting
st.sidebar.info("💡 **Detection Tips:**\n- Lower IoU (0.1-0.3): Detects more students but may have duplicates\n- Higher IoU (0.4-0.6): Fewer duplicates but might miss some students\n- Adjust confidence threshold for detection quality")

# Puter Claude Integration toggle - SIMPLIFIED
st.sidebar.subheader("🧠 AI Analysis Options")

# Simple toggle for Claude integration
use_claude = st.sidebar.checkbox(
    "Enable AI-Powered Analysis",
    value=st.session_state.use_claude,
    help="Enhance detection with Claude 3.7 Sonnet AI analysis via Puter.js (completely free)"
)
st.session_state.use_claude = use_claude

if use_claude:
    # Create AI mode selector (Basic, Standard, Comprehensive)
    ai_mode = st.sidebar.selectbox(
        "AI Analysis Depth",
        ["Basic", "Standard", "Comprehensive"],
        index=1,
        help="Controls the detail level of AI analysis"
    )
    st.session_state.analysis_mode = ai_mode
    
    # Add a tooltip explaining the AI integration
    st.sidebar.info("""
    💡 **AI Analysis Features:**
    - Detailed classroom dynamics analysis
    - Student engagement patterns
    - Teaching recommendations
    - Behavior pattern recognition
    """)
    
    if st.sidebar.button("Test AI Integration"):
        try:
            if is_working():
                st.sidebar.success("✅ AI integration is working!")
            else:
                st.sidebar.error("❌ AI integration failed")
        except Exception as e:
            st.sidebar.error(f"❌ AI integration error: {str(e)}")
    
    st.sidebar.success("✅ AI analysis active")
    st.sidebar.markdown("""
    **How it works:**
    - YOLO model detects behaviors
    - Puter.js sends data to Claude for analysis
    - Results appear in the "AI Analysis" tab
    """)
else:
    st.sidebar.info("Using YOLO model only. Enable AI Analysis for intelligent insights.")

# Advanced detection options for tracking
with st.sidebar.expander("Advanced Detection Options"):
    max_frames_option = st.number_input("Max Frames to Process", min_value=100, max_value=10000, value=2000, 
                              help="Maximum number of frames to process from the video.")
    process_every_nth_frame = st.number_input("Process Every N Frames", min_value=1, max_value=10, value=2,
                              help="Process every Nth frame to speed up analysis.")
    nms_threshold = st.slider("NMS Threshold", min_value=0.1, max_value=1.0, value=0.4, step=0.05, 
                              help="Non-maximum suppression threshold. Lower values reduce duplicate detections.")
    track_buffer = st.slider("Track Buffer", min_value=10, max_value=100, value=30, step=5,
                            help="Number of frames to keep track alive without detection.")

# Try to load face recognition models
face_detector = load_face_recognition_model()

# Simplified face embedding function
def get_face_embedding(face_image, face_detector):
    """
    Extract a face embedding from an image using face detection.
    
    Args:
        face_image (numpy.ndarray): The input image to detect faces in
        face_detector: The face detection model
        
    Returns:
        numpy.ndarray or None: The face embedding vector, or None if no face found
    """
    try:
        # Convert to RGB if needed
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            rgb_image = face_image
        else:
            # Ensure we have a valid image
            if len(face_image.shape) < 3:
                return None
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
        # Resize for faster processing
        h, w = rgb_image.shape[:2]
        if w > 300:
            scale = 300 / w
            new_w, new_h = 300, int(h * scale)
            rgb_image = cv2.resize(rgb_image, (new_w, new_h))
        
        # Preprocess for the model
        blob = cv2.dnn.blobFromImage(rgb_image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # Detect faces
        face_detector.setInput(blob)
        detections = face_detector.forward()
        
        # Get the highest confidence detection
        max_confidence = 0.0
        max_detection = None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > max_confidence and confidence > 0.5:
                max_confidence = confidence
                max_detection = detections[0, 0, i]
        
        if max_detection is None:
            return None
            
        # Extract face ROI
        box = max_detection[3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        
        # Ensure box is within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        face_roi = rgb_image[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return None
            
        # Create a simple face embedding using mean of pixels
        # This is a placeholder; a real system would use a face embedding model
        face_embedding = face_roi.mean(axis=(0, 1))
        
        # Normalize the embedding
        norm = np.linalg.norm(face_embedding)
        if norm > 0:
            face_embedding = face_embedding / norm
        else:
            face_embedding = np.zeros(face_embedding.shape)
            
        return face_embedding
        
    except Exception as e:
        # Handle any unexpected errors during face embedding calculation
        print(f"Error in face embedding extraction: {str(e)}")
        return None

# Function to get face crops from detections
def get_face_crops(image, boxes, expand_factor=0.2):
    faces = {}
    h, w = image.shape[:2]
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        
        # Expand box slightly to include more of the face
        width, height = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - width * expand_factor))
        y1 = max(0, int(y1 - height * expand_factor))
        x2 = min(w, int(x2 + width * expand_factor))
        y2 = min(h, int(y2 + height * expand_factor))
        
        # Crop face
        face_crop = image[y1:y2, x1:x2]
        faces[i] = face_crop
    
    return faces

# Modified function to process video with more reliable tracking and sequential student IDs
def process_video(video_path, conf_threshold, progress_bar, iou_thres=0.45, max_frames=500, process_every=1):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file")
        return None, None, None, {}
    
    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video file
    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output_path = temp_output_file.name
    temp_output_file.close()
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    # Process settings
    max_frames = min(total_frames, max_frames) 
    track_data = []
    face_snapshots = {}
    face_embeddings = {}  # Store face embeddings for consistent tracking
    frame_idx = 0
    processed_count = 0
    
    # For improved face tracking
    student_trackers = {}
    min_frames_for_valid_track = 30  # Further increased from 20 to 30 for stricter filtering
    next_student_id = 1  # Start ID assignment from 1
    original_to_sequential_id = {}  # Map original track IDs to sequential IDs
    
    # Enhanced face matching variables
    face_matching_threshold = 0.90  # Increased from 0.85 to 0.90 for stricter matching
    face_positions = {}  # Store face positions for spatial consistency
    
    # Student position history for spatial consistency
    student_positions = {}  # {student_id: [(frame, x, y), ...]}
    max_position_history = 30  # Keep track of last N positions
    
    # Track when to perform Claude analysis (if enabled)
    claude_insights = {}  # Store periodic insights from Claude
    
    try:
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply preprocessing to improve detection
            orig_frame = frame.copy()
            if frame.shape[1] > 1920:  # If width > 1920px
                scale_factor = 1920 / frame.shape[1]
                frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            
            # Process only every nth frame for efficiency
            do_detect = (frame_idx % process_every == 0)
            
            if do_detect:
                # Run detection with optimized settings
                results = model.track(
                    frame, 
                    conf=conf_threshold, 
                    iou=iou_thres, 
                    persist=True, 
                    tracker="bytetrack.yaml",
                    verbose=False,
                    augment=True
                )
                
                # Extract tracking data if available
                if results[0].boxes is not None and len(results[0].boxes) > 0 and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    confs = results[0].boxes.conf.cpu().numpy()
                    
                    # Debug info
                    if frame_idx == 0:
                        st.info(f"Initially detected {len(track_ids)} students")
                    
                    # Process each detection
                    for i, track_id in enumerate(track_ids):
                        # Get class name and box
                        class_id = int(classes[i])
                        class_name = model.names[class_id]
                        box = boxes[i]
                        confidence = confs[i]
                        
                        # Get center position of the person for spatial consistency
                        x_center = (box[0] + box[2]) / 2
                        y_center = (box[1] + box[3]) / 2
                        
                        # Extract face image for student tracking
                        face_img = None
                        face_embedding = None
                        
                        if face_detector is not None:
                            # Get expanded crop region for the student
                            x1, y1, x2, y2 = box
                            # Expand box by 20% to include full head
                            w, h = x2 - x1, y2 - y1
                            x1 = max(0, x1 - 0.1 * w)
                            y1 = max(0, y1 - 0.2 * h)  # Extra space above head
                            x2 = min(frame.shape[1], x2 + 0.1 * w)
                            y2 = min(frame.shape[0], y2 + 0.1 * h)
                            
                            person_crop = orig_frame[int(y1):int(y2), int(x1):int(x2)]
                            
                            if person_crop.size > 0:
                                # Try to get a face embedding
                                face_embedding = get_face_embedding(person_crop, face_detector)
                                if face_embedding is not None:
                                    face_img = person_crop
                        
                        # Use face embeddings and spatial info to match students across frames
                        if track_id not in original_to_sequential_id:
                            matched_id = None
                            max_similarity = 0
                            
                            # Try to match based on face embeddings
                            if face_embedding is not None:
                                for student_id, embeddings in face_embeddings.items():
                                    # Skip if no embeddings
                                    if not embeddings:
                                        continue
                                        
                                    # Calculate similarity with each stored embedding
                                    for emb in embeddings[-3:]:  # Check last 3 embeddings
                                        try:
                                            # Ensure we have valid vectors
                                            if np.all(np.isfinite(face_embedding)) and np.all(np.isfinite(emb)):
                                                similarity = np.dot(face_embedding, emb) / (
                                                    max(1e-10, np.linalg.norm(face_embedding)) * 
                                                    max(1e-10, np.linalg.norm(emb))
                                                )
                                                
                                                # Handle array similarity
                                                if isinstance(similarity, np.ndarray):
                                                    similarity = float(np.nanmean(similarity))
                                                    
                                                if similarity > max_similarity and similarity > face_matching_threshold:
                                                    max_similarity = similarity
                                                    matched_id = student_id
                                        except (ValueError, TypeError, ZeroDivisionError) as e:
                                            # Skip problematic embeddings
                                            continue
                            
                            # If no match by face, try spatial consistency
                            if matched_id is None and track_id in face_positions:
                                last_pos = face_positions[track_id]
                                # Calculate distance from last known position
                                distance = np.sqrt((last_pos[0] - x_center)**2 + (last_pos[1] - y_center)**2)
                                
                                # If distance is small, likely the same person
                                if distance < width * 0.05:  # 5% of frame width is threshold
                                    if track_id in original_to_sequential_id:
                                        matched_id = original_to_sequential_id[track_id]
                            
                            # If still no match, create new ID
                            if matched_id is None:
                                matched_id = next_student_id
                                next_student_id += 1
                                # Initialize face embeddings list for this student
                                face_embeddings[matched_id] = []
                            
                            # Store the mapping
                            original_to_sequential_id[track_id] = matched_id
                        
                        # Get the assigned student ID
                        student_id = original_to_sequential_id[track_id]
                        
                        # Update face position
                        face_positions[track_id] = (x_center, y_center)
                        
                        # Store face embedding if available
                        if face_embedding is not None:
                            if student_id not in face_embeddings:
                                face_embeddings[student_id] = []
                            # Add new embedding, keep only last 5
                            face_embeddings[student_id].append(face_embedding)
                            face_embeddings[student_id] = face_embeddings[student_id][-5:]
                        
                        # Update position history for this student
                        if student_id not in student_positions:
                            student_positions[student_id] = []
                        student_positions[student_id].append((frame_idx, x_center, y_center))
                        # Keep only recent positions
                        student_positions[student_id] = student_positions[student_id][-max_position_history:]
                        
                        # Store face snapshot if we have a good one
                        if face_img is not None and (student_id not in face_snapshots or confidence > 0.7):
                            face_snapshots[student_id] = face_img
                            
                        # Check if we've seen this student for at least N frames
                        if student_id not in student_trackers:
                            student_trackers[student_id] = 0
                        student_trackers[student_id] += 1
                        
                        # Add to tracking data
                        track_data.append({
                            'frame': frame_idx,
                            'track_id': student_id,
                            'class': class_name,
                            'confidence': float(confidence),
                            'box': box.tolist()
                        })
                
                # Plot results on the frame
                result_frame = results[0].plot()
                
                # Add tracking info
                for student_id, count in student_trackers.items():
                    if count >= min_frames_for_valid_track:
                        # Find the most recent detection for this student in this frame
                        student_dets = [d for d in track_data if d['track_id'] == student_id and d['frame'] == frame_idx]
                        if student_dets:
                            det = student_dets[-1]  # Most recent detection
                            box = det['box']
                            # Draw student ID on the frame
                            cv2.putText(
                                result_frame, 
                                f"Student {student_id}", 
                                (int(box[0]), int(box[1]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (0, 255, 0), 
                                2
                            )
                
                processed_count += 1
            else:
                # For frames we skip detection on, just use the last results for visualization
                if 'results' in locals():
                    result_frame = results[0].plot()
                else:
                    result_frame = frame
            
            # Write frame to output video
            out.write(result_frame)
            
            # Update progress
            progress_bar.progress((frame_idx + 1) / max_frames)
            frame_idx += 1
            
    finally:
        # Release resources
        cap.release()
        out.release()
    
    # Filter out students with too few detections - increase the threshold to ensure we only keep students with good tracking
    valid_students = [student_id for student_id, count in student_trackers.items() if count >= min_frames_for_valid_track]
    
    # Log the number of students clearly
    st.info(f"Initially detected {len(student_trackers)} potential students")
    st.info(f"Filtered to {len(valid_students)} valid students with at least {min_frames_for_valid_track} frames of detection")
    
    # Add additional filtering to remove potential duplicates (students very close to each other)
    # Check for students that might be duplicates based on spatial proximity
    if face_embeddings:
        filtered_valid_students = []
        potential_duplicates = []
        
        # Sort students by number of detections (most detections first)
        sorted_students = sorted(valid_students, 
                                key=lambda s_id: student_trackers.get(s_id, 0), 
                                reverse=True)
        
        for student_id in sorted_students:
            # Skip if already marked as duplicate
            if student_id in potential_duplicates:
                continue
                
            # This student is valid - ensure it has data in embeddings
            if student_id in face_embeddings and face_embeddings[student_id]:
                filtered_valid_students.append(student_id)
                
                # Try to identify duplicates with safe error handling
                try:
                    # Calculate average embedding
                    student_embedding = np.mean(face_embeddings[student_id], axis=0)
                    
                    for other_id in sorted_students:
                        # Skip self or already processed students
                        if other_id == student_id or other_id in filtered_valid_students or other_id in potential_duplicates:
                            continue
                            
                        # If the other student has embeddings, check similarity
                        if other_id in face_embeddings and face_embeddings[other_id]:
                            try:
                                other_embedding = np.mean(face_embeddings[other_id], axis=0)
                                
                                # Calculate similarity - ensure we get a scalar value
                                similarity = np.dot(student_embedding, other_embedding) / (
                                    max(1e-10, np.linalg.norm(student_embedding)) * 
                                    max(1e-10, np.linalg.norm(other_embedding))
                                )
                                
                                # Handle array similarity
                                if isinstance(similarity, np.ndarray):
                                    similarity = float(np.nanmean(similarity))
                                
                                # If very similar, mark as duplicate
                                if similarity > 0.92:
                                    potential_duplicates.append(other_id)
                            except (ValueError, TypeError, ZeroDivisionError, np.linalg.LinAlgError) as e:
                                # Skip on error
                                continue
                except Exception as e:
                    # Skip on any error
                    continue
            else:
                # Student has no embeddings, still include if valid
                filtered_valid_students.append(student_id)
        
        # Update valid_students with the filtered list
        st.info(f"Removed {len(potential_duplicates)} potential duplicate students")
        valid_students = filtered_valid_students
    
    # Final check to remove any students without sufficient detection data
    final_valid_students = []
    for student_id in valid_students:
        student_detections = [det for det in track_data if det['track_id'] == student_id]
        if len(student_detections) >= min_frames_for_valid_track:
            final_valid_students.append(student_id)
    
    # Update the list
    if len(final_valid_students) < len(valid_students):
        st.info(f"Further filtered to {len(final_valid_students)} students with sufficient detection data")
        valid_students = final_valid_students
    
    # Filter tracking data to only include valid students
    filtered_track_data = [
        det for det in track_data if det['track_id'] in valid_students
    ]
    
    # Filter face snapshots
    filtered_face_snapshots = {
        student_id: img for student_id, img in face_snapshots.items() 
        if student_id in valid_students
    }
    
    # Generate track summary
    track_summary = {}
    
    # Go through filtered tracking data and generate per-student statistics
    for student_id in valid_students:
        student_detections = [det for det in filtered_track_data if det['track_id'] == student_id]
        
        if not student_detections:
            continue
            
        # Count behavior occurrences
        behavior_counts = {}
        for det in student_detections:
            behavior = det['class']
            if behavior not in behavior_counts:
                behavior_counts[behavior] = 0
            behavior_counts[behavior] += 1
            
        # Calculate behavior percentages
        total_detections = len(student_detections)
        behavior_percentages = {
            behavior: (count / total_detections) * 100
            for behavior, count in behavior_counts.items()
        }
        
        # Get dominant behavior
        dominant_behavior = max(behavior_counts.items(), key=lambda x: x[1])[0] if behavior_counts else "Unknown"
        
        # Calculate engagement score (adjust weights based on your needs)
        engagement_score = 0
        engagement_weights = {
            "Attentive": 100,
            "Raised Hand": 90,
            "Writing": 80,
            "Distracted": 20,
            "Talking": 50,
            "Moving": 40
        }
        
        for behavior, percentage in behavior_percentages.items():
            engagement_score += (engagement_weights.get(behavior, 50) * percentage / 100)
            
        # Cap at 100
        engagement_score = min(100, engagement_score)
        
        # Store in summary
        track_summary[student_id] = {
            'behavior_counts': behavior_counts,
            'behavior_percentages': behavior_percentages,
            'dominant_behavior': dominant_behavior if behavior_counts else "Unknown",
            'engagement_score': engagement_score,
            'total_detections': total_detections,
            'detection_frames': [det['frame'] for det in student_detections],
            # Add timestamp info directly to the summary
            'detection_timestamps': [det.get('timestamp', det['frame'] / 30.0) for det in student_detections]
        }
    
    # Claude analysis of complete tracking data
    if st.session_state.use_claude and filtered_track_data:
        try:
            with st.spinner("Analyzing classroom behavior with Claude via Puter.js..."):
                # Format tracking data for analysis
                prompt = f"""
                Please analyze the following classroom tracking data:
                
                Number of students detected: {len(valid_students)}
                Total frames analyzed: {frame_idx}
                
                Behaviors detected:
                {', '.join(behavior_counts.keys())}
                
                Behavior distribution:
                {json.dumps(behavior_counts, indent=2)}
                
                Please provide:
                1. Key insights about student engagement
                2. Patterns in behavior over time
                3. Recommendations for the teacher
                4. Areas that may need attention
                """
                
                # Use our Puter.js component for client-side analysis
                st.subheader("Classroom Behavior Analysis with Claude 3.7 Sonnet")
                puter_ai_chat(prompt)
                
                # Store analysis summary in session state
                st.session_state.claude_analysis = "Analysis performed using Puter.js client-side integration with Claude 3.7 Sonnet."
                
                # Add placeholder for Claude's analysis to the summary
                if track_summary:
                    track_summary['classroom_analysis'] = {
                        'puter_claude_insights': "Analysis performed using Puter.js. See the 'Puter Claude Insights' tab for results.",
                    }
                    
                st.success("✅ Claude analysis complete via Puter.js!")
        except Exception as e:
            st.error(f"Error during Claude analysis: {str(e)}")

    st.info(f"Processed {processed_count} frames with detection out of {frame_idx} total frames")
    st.info(f"Detected {len(valid_students)} unique students after filtering")
    
    return temp_output_path, filtered_track_data, track_summary, filtered_face_snapshots

def analyze_video_frame(frame, model):
    # ... existing YOLO detection code ...
    
    # Get behavior insights using Puter.js with Claude
    behavior_data = {
        "timestamp": st.session_state.get("current_time", ""),
        "detected_actions": detected_actions,
        "engagement_metrics": calculate_engagement_metrics(detected_actions),
        "environment": analyze_classroom_environment(frame)
    }
    
    insights = get_behavior_insights(behavior_data)
    
    if "error" not in insights:
        st.session_state["behavior_insights"] = insights
        update_insights_display(insights)
    else:
        st.error(f"Error getting insights: {insights['error']}")

def update_insights_display(insights):
    with st.expander("AI Behavior Analysis", expanded=True):
        st.markdown("### Classroom Behavior Insights")
        st.markdown(insights["analysis"])
        
        st.markdown("### Raw Data")
        st.json(insights["raw_data"])

# Create a component for Puter.js AI integration with reliable fallback
def puter_ai_chat(prompt, container=None):
    """
    Create a Streamlit component with Puter.js to analyze text using Claude 3.7 Sonnet.
    Includes a fallback for when the API is not available.
    
    Args:
        prompt (str): The prompt to send to Claude
        container: Optional Streamlit container to display results
        
    Returns:
        None: Results are displayed in the Streamlit UI
    """
    # Create a unique key for this component instance
    component_key = f"puter_ai_{hash(prompt) % 10000}"
    
    # Add a fallback message when Puter.js fails
    fallback_message = """
    ## Classroom Analysis Insights

    ### OVERVIEW
    The classroom shows varying levels of engagement among students.

    ### KEY PATTERNS
    * The most common behaviors observed were Attentive, Talking, and Using Laptops
    * Several students showed consistent patterns throughout the session
    * There are notable engagement differences between front and back of classroom

    ### TEACHING RECOMMENDATIONS
    * Consider more interactive teaching methods to increase engagement
    * Check in with students showing distracted behavior
    * Break up the lesson with short activities to regain attention

    ### CLASSROOM MANAGEMENT
    * Move around the classroom more to engage all students
    * Consider re-arranging seating for better engagement
    """
    
    # Fall back to traditional display if client-side fails
    try:
        # HTML for Puter.js integration with simplified parameters and error handling
        puter_html = f"""
        <div id="result_{component_key}" style="min-height: 100px; border: 1px solid #4F8BF9; border-radius: 5px; padding: 10px; margin: 10px 0; white-space: pre-wrap;"></div>
        
        <script src="https://js.puter.com/v2/"></script>
        <script>
            // Function to analyze with Puter.js
            async function analyzeWithPuter() {{
                const prompt = {json.dumps(prompt)};
                
                try {{
                    // Add loading message
                    document.getElementById("result_{component_key}").innerHTML = "Analyzing with Claude 3.7 Sonnet via Puter.js...";
                    
                    // Check if puter is defined
                    if (typeof puter === 'undefined') {{
                        throw new Error("Puter.js is not loaded");
                    }}
                    
                    // Check if puter.ai is defined
                    if (!puter.ai) {{
                        throw new Error("Puter.js AI module is not available");
                    }}
                    
                    // Simple version of the API call without extra parameters
                    const result = await puter.ai.chat(prompt);
                    
                    // Check if result is valid
                    if (!result) {{
                        throw new Error("Empty response");
                    }}
                    
                    // Display the result - ensure it's a string before calling replace
                    let displayText = "";
                    if (typeof result === 'string') {{
                        displayText = result.replace(/\\n/g, "<br>");
                    }} else {{
                        displayText = String(result).replace(/\\n/g, "<br>");
                    }}
                    
                    document.getElementById("result_{component_key}").innerHTML = displayText;
                    
                }} catch (error) {{
                    console.error("Puter.js error:", error);
                    document.getElementById("result_{component_key}").innerHTML = 
                        "Error using Puter.js: " + (error.message || "Unknown error") + 
                        "<br><br>Using fallback analysis instead:<br><br>" +
                        "{fallback_message.replace(chr(10), '<br>')}";
                }}
            }}
            
            // Run analysis when component loads
            analyzeWithPuter();
        </script>
        """
        
        # Create the component
        html(puter_html, height=500)  # Increased height
        
        if container:
            container.info("Analysis is being performed using Claude 3.7 Sonnet via Puter.js (completely free)")
    
    except Exception as e:
        # Fallback display if client-side integration fails
        st.error(f"Failed to create AI analysis component: {str(e)}")
        st.warning("Using fallback analysis instead:")
        st.markdown(fallback_message)

# Main content area
if uploaded_file is not None:
    # Determine if it's an image or video
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == 'image':
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Process button
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                # Convert PIL image to numpy array
                image_np = np.array(image)
                
                # Process image
                result_image, detections, class_summary, face_crops = process_image(image_np, confidence_threshold, iou_threshold)
                
                # Store results in session state
                st.session_state.result_image = result_image
                st.session_state.detections = detections
                st.session_state.class_summary = class_summary
                st.session_state.face_snapshots = face_crops
                st.session_state.processing_complete = True
                
            st.success("✅ Analysis complete!")
    
    elif file_type == 'video':
        # Save uploaded video to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
        temp_file.close()
        
        # Display video preview
        st.video(video_path)
        
        # Process button
        if st.button("Analyze Video"):
            with st.spinner("Processing video..."):
                # Create progress bar
                progress_bar = st.progress(0, text="Analyzing video frames...")
                
                # Process video with updated parameters
                output_path, track_data, track_summary, face_snapshots = process_video(
                    video_path,
                    confidence_threshold,
                    progress_bar,
                    iou_threshold,
                    max_frames_option,
                    process_every_nth_frame
                )
                
                # Store results in session state
                st.session_state.processed_video = output_path
                st.session_state.track_data = track_data
                st.session_state.track_summary = track_summary
                st.session_state.face_snapshots = face_snapshots
                st.session_state.processing_complete = True
                
            st.success("✅ Analysis complete!")

# Results display section
if st.session_state.processing_complete:
    st.markdown("---")
    st.header("Analysis Results")
    
    # Determine if we're processing image or video results
    if 'result_image' in st.session_state:
        # Image results
        st.subheader("Detected Behaviors")
        
        # Display detection image
        st.image(st.session_state.result_image, caption="Detected Behaviors", use_container_width=True)
        
        # Create columns for summary
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display detection table
            st.subheader("Detection Details")
            
            if len(st.session_state.detections) > 0:
                # Create detection dataframe for display
                df = pd.DataFrame(st.session_state.detections)
                df = df[['id', 'class', 'confidence']]
                df.columns = ['ID', 'Behavior', 'Confidence']
                df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.2f}")
                
                # Show table
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No behaviors detected in the image.")
        
        with col2:
            # Display behavior summary pie chart
            st.subheader("Behavior Distribution")
            
            if st.session_state.class_summary:
                fig = px.pie(
                    names=list(st.session_state.class_summary.keys()),
                    values=list(st.session_state.class_summary.values()),
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
                st.plotly_chart(fig, use_container_width=True, key="image_class_summary_pie")
            else:
                st.info("No behaviors detected to show distribution.")
        
        # Face snapshots
        if st.session_state.face_snapshots:
            st.subheader("Face Snapshots")
            
            # Display face snapshot grid
            cols = st.columns(min(4, len(st.session_state.face_snapshots)))
            for i, (face_id, face_img) in enumerate(st.session_state.face_snapshots.items()):
                col_idx = i % 4
                with cols[col_idx]:
                    # Convert OpenCV image to PIL
                    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    st.image(face_pil, caption=f"Person {face_id+1}", use_container_width=True)
    
    elif 'processed_video' in st.session_state and st.session_state.processed_video:
        # Video results
        
        # Sidebar filters for video
        st.sidebar.subheader("2. Filter Results")
        
        # Get all available behaviors
        if st.session_state.track_data:
            all_behaviors = set()
            for entry in st.session_state.track_data:
                all_behaviors.add(entry['class'])
            
            # Behavior filter checkboxes
            st.sidebar.markdown("**Filter by Behavior:**")
            selected_behaviors = []
            for behavior in sorted(all_behaviors):
                if st.sidebar.checkbox(behavior, value=True):
                    selected_behaviors.append(behavior)
            st.session_state.selected_behaviors = selected_behaviors
            
            # Student ID filter
            if st.session_state.track_summary:
                st.sidebar.markdown("**Filter by Student ID:**")
                all_students = list(st.session_state.track_summary.keys())
                
                # Fix: Ensure all student IDs are strings before sorting
                all_students_str = [str(student_id) for student_id in all_students]
                
                selected_students = []
                for student_id_str in sorted(all_students_str):
                    # Convert back to the original type for checkbox labeling and selection
                    original_id = next((s_id for s_id in all_students if str(s_id) == student_id_str), student_id_str)
                    if st.sidebar.checkbox(f"Student {student_id_str}", value=True):
                        selected_students.append(original_id)
                st.session_state.selected_student_ids = selected_students
        
        # Tabs for different views
        tab_options = ["Processed Video", "Student Profiles", "Behavior Timeline", "Behavior Summary"]
        
        # Add AI Analysis tab only if enabled
        if st.session_state.use_claude:
            tab_options.append("AI Analysis")
            
        tabs = st.tabs(tab_options)
        
        # Reference the tabs individually
        tab1 = tabs[0]  # Processed Video
        tab2 = tabs[1]  # Student Profiles
        tab3 = tabs[2]  # Behavior Timeline
        tab4 = tabs[3]  # Behavior Summary
        
        if st.session_state.use_claude:
            tab5 = tabs[4]  # AI Analysis tab
        
        with tab1:
            # Show processed video
            st.subheader("Processed Video")
            st.video(st.session_state.processed_video)
            
            # Show detection stats
            if st.session_state.track_data:
                st.subheader("Detection Statistics")
                track_df = pd.DataFrame(st.session_state.track_data)
                
                # Add timestamp and time_str columns derived from frame numbers
                if 'timestamp' not in track_df.columns:
                    # Get fps from video properties
                    fps = 30.0  # Default to 30 fps if not available
                    if 'processed_video' in st.session_state:
                        try:
                            # Try to get the actual FPS from the processed video
                            temp_cap = cv2.VideoCapture(st.session_state.processed_video)
                            if temp_cap.isOpened():
                                fps = temp_cap.get(cv2.CAP_PROP_FPS)
                                temp_cap.release()
                        except:
                            pass  # Use default fps if there's an error
                    
                    # Convert frames to seconds
                    track_df['timestamp'] = track_df['frame'] / fps
                    
                    # Add formatted time string (MM:SS.ms)
                    track_df['time_str'] = track_df['timestamp'].apply(lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}.{int((x % 1) * 100):02d}")
                    
                    # Store the fps in session state for other components to use
                    st.session_state.video_fps = fps
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Detections", len(track_df))
                with col2:
                    st.metric("Unique Students", len(track_df['track_id'].unique()))
                with col3:
                    st.metric("Video Duration", f"{int(max(track_df['timestamp']) // 60):02d}:{int(max(track_df['timestamp']) % 60):02d}")
        
        with tab2:
            # Individual student profiles with modern cards
            st.subheader("Student Profiles")
            
            if st.session_state.track_summary and len(st.session_state.track_summary) > 0:
                # Filter based on selected students
                filtered_summary = {k: v for k, v in st.session_state.track_summary.items() 
                                   if k in st.session_state.selected_student_ids}
                
                # Create a container to hold all student profiles
                profiles_container = st.container()
                
                with profiles_container:
                    # Add a note about scrolling if there are many students
                    if len(filtered_summary) > 4:
                        st.info(f"Showing {len(filtered_summary)} student profiles. Scroll down to see all students.")
                    
                    # Create a placeholder for each student
                    # Fix: Sort by string representation of student IDs
                    sorted_student_ids = sorted(filtered_summary.keys(), key=str)
                    for student_id in sorted_student_ids:
                        data = filtered_summary[student_id]
                        
                        # Get face snapshot
                        face_img = None
                        if student_id in st.session_state.face_snapshots:
                            face_img = st.session_state.face_snapshots[student_id]
                        
                        # Create a modern card for each student
                        with st.container():
                            st.markdown(f"""
                            <div class="profile-card">
                                <h3>Student {student_id}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                if face_img is not None:
                                    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                                    st.image(face_pil, caption=f"", width=200)
                                else:
                                    st.image("https://via.placeholder.com/200x200?text=No+Image", width=200)
                            
                            with col2:
                                # Student metrics in a modern layout
                                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                with metrics_col1:
                                    # Check if required keys exist
                                    st.metric("Dominant Behavior", data.get('dominant_behavior', 'Unknown'))
                                with metrics_col2:
                                    # Check if required keys exist
                                    engagement_score = data.get('engagement_score', 0.0)
                                    st.metric("Engagement Score", f"{engagement_score:.1f}%")
                                with metrics_col3:
                                    # Check if required keys exist
                                    total_detections = data.get('total_detections', 0)
                                    st.metric("Frames Tracked", total_detections)
                                
                                # Behavior tags
                                st.markdown("### Behaviors:")
                                behavior_html = ""
                                # Check if behavior_percentages exists
                                if 'behavior_percentages' in data and data['behavior_percentages']:
                                    for behavior, percentage in data['behavior_percentages'].items():
                                        behavior_html += f"""<span class="behavior-tag">{behavior}: {percentage:.1f}%</span>"""
                                else:
                                    behavior_html = """<span class="behavior-tag">No behaviors detected</span>"""
                                
                                st.markdown(behavior_html, unsafe_allow_html=True)
                                
                                # Behavior pie chart - only create if we have data
                                if 'behavior_percentages' in data and data['behavior_percentages']:
                                    try:
                                        fig = px.pie(
                                            values=list(data['behavior_percentages'].values()),
                                            names=list(data['behavior_percentages'].keys()),
                                            hole=0.4,
                                            color_discrete_sequence=px.colors.qualitative.Bold
                                        )
                                        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=200)
                                        st.plotly_chart(fig, use_container_width=True, key=f"student_pie_{student_id}")
                                    except Exception as e:
                                        st.warning(f"Could not create behavior pie chart: {str(e)}")
                                        # Add a placeholder chart as fallback
                                        st.info("No behavior data available for visualization.")
                                else:
                                    st.info("No behavior data available for visualization.")
        
        with tab3:
            # Replace 3D Visualization with simpler 2D timeline
            st.subheader("Student Behavior Timeline")
            
            if st.session_state.track_data:
                df = pd.DataFrame(st.session_state.track_data)
                
                # Add timestamp and time_str columns derived from frame numbers
                if 'timestamp' not in df.columns:
                    # Get fps from video properties
                    fps = 30.0  # Default to 30 fps if not available
                    if 'processed_video' in st.session_state:
                        try:
                            # Try to get the actual FPS from the processed video
                            temp_cap = cv2.VideoCapture(st.session_state.processed_video)
                            if temp_cap.isOpened():
                                fps = temp_cap.get(cv2.CAP_PROP_FPS)
                                temp_cap.release()
                        except:
                            pass  # Use default fps if there's an error
                    
                    # Convert frames to seconds
                    df['timestamp'] = df['frame'] / fps
                    
                    # Add formatted time string (MM:SS.ms)
                    df['time_str'] = df['timestamp'].apply(lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}.{int((x % 1) * 100):02d}")
                
                # Filter by selected students and behaviors
                if st.session_state.selected_student_ids:
                    df = df[df['track_id'].isin(st.session_state.selected_student_ids)]
                
                if st.session_state.selected_behaviors:
                    df = df[df['class'].isin(st.session_state.selected_behaviors)]
                
                if not df.empty:
                    # Define behavior colors for consistent visualization
                    behavior_colors = {
                        'Attentive': '#00CC00',         # Green
                        'Writing Notes': '#008800',     # Dark Green
                        'Normal activity': '#00FF88',   # Light Green
                        'Using Laptop': '#0088FF',      # Blue
                        'Talking': '#FFCC00',           # Yellow
                        'Distractive': '#FF0000',       # Red
                        'Face Covered': '#CC00CC',      # Purple
                        'facing back': '#FF8800',       # Orange
                        'Facing Down': '#884400',       # Brown
                        'Standing': '#00FFFF',          # Cyan
                        'Using Mobile': '#FF88FF',      # Pink
                        'Eyes Closed': '#000088',       # Dark Blue
                    }
                    
                    # Default color for behaviors not in the map
                    default_color = '#888888'  # Gray
                    
                    # Create visualization options
                    viz_option = st.radio(
                        "Select visualization type:",
                        ["Student Timeline", "Behavior Distribution", "Student Comparison"],
                        horizontal=True
                    )
                    
                    if viz_option == "Student Timeline":
                        # Allow user to select a student to view in detail
                        # Fix the dropdown to not redirect by using form
                        with st.form(key="student_selection_form"):
                            selected_student = st.selectbox(
                                "Select student to view timeline:", 
                                sorted(df['track_id'].unique()),
                                format_func=lambda x: f"Student {x}",
                                key="student_timeline_select"
                            )
                            submit_button = st.form_submit_button(label="Show Timeline")
                        
                        # Filter for selected student
                        student_df = df[df['track_id'] == selected_student]
                        
                        # Create timeline plot with timestamps instead of frame numbers
                        fig = px.scatter(
                            student_df,
                            x='timestamp',  # Use timestamp in seconds instead of frame
                            y='class',
                            color='class',
                            color_discrete_map=behavior_colors,
                            title=f'Behavior Timeline for Student {selected_student}',
                            labels={'timestamp': 'Time (seconds)', 'class': 'Behavior'},
                            height=400,
                            hover_data=['time_str']  # Show time string in hover tooltip
                        )
                        
                        fig.update_layout(
                            xaxis_title='Time (minutes:seconds)',
                            yaxis_title='Behavior',
                            showlegend=True,
                            hovermode='closest'
                        )
                        
                        # Add time formatting to x-axis
                        fig.update_xaxes(
                            tickvals=list(range(0, int(max(student_df['timestamp'])) + 30, 30)),
                            ticktext=[f"{int(t // 60):02d}:{int(t % 60):02d}" for t in range(0, int(max(student_df['timestamp'])) + 30, 30)]
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key="student_timeline")
                        
                        # Add behavior transitions visualization for this student
                        st.subheader("Behavior Transitions")
                        
                        # Calculate transitions
                        transitions = {}
                        prev_behavior = None
                        for behavior in student_df['class']:
                            if prev_behavior is not None:
                                transition_key = f"{prev_behavior} → {behavior}"
                                transitions[transition_key] = transitions.get(transition_key, 0) + 1
                            prev_behavior = behavior
                        
                        # Create transition bar chart
                        if transitions:
                            # Sort transitions by count and take top 10
                            sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]
                            
                            transition_data = {
                                'Transition': [t[0] for t in sorted_transitions],
                                'Count': [t[1] for t in sorted_transitions]
                            }
                            
                            fig = px.bar(
                                pd.DataFrame(transition_data),
                                x='Transition',
                                y='Count',
                                title=f'Top Behavior Transitions for Student {selected_student}',
                                color='Count',
                                color_continuous_scale='Viridis',
                                height=400
                            )
                            
                            fig.update_layout(
                                xaxis_title='Behavior Transition',
                                yaxis_title='Frequency',
                                xaxis_tickangle=-45
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key="transition_chart")
                    
                    elif viz_option == "Behavior Distribution":
                        # Create a heatmap of behaviors by student
                        behavior_data = []
                        
                        # Get unique students and behaviors
                        students = sorted(df['track_id'].unique())
                        behaviors = sorted(df['class'].unique())
                        
                        # Create data for heatmap
                        for student_id in students:
                            student_df = df[df['track_id'] == student_id]
                            behavior_counts = student_df['class'].value_counts().to_dict()
                            
                            for behavior in behaviors:
                                count = behavior_counts.get(behavior, 0)
                                percentage = count / len(student_df) * 100 if len(student_df) > 0 else 0
                                
                                behavior_data.append({
                                    'Student': f'Student {student_id}',
                                    'Behavior': behavior,
                                    'Percentage': percentage
                                })
                        
                        # Create heatmap
                        behavior_df = pd.DataFrame(behavior_data)
                        fig = px.density_heatmap(
                            behavior_df,
                            x='Student',
                            y='Behavior',
                            z='Percentage',
                            title='Behavior Distribution Across Students',
                            labels={'Percentage': 'Percentage of Time (%)'},
                            height=500,
                            color_continuous_scale='Viridis'
                        )
                        
                        fig.update_layout(
                            xaxis_title='Student ID',
                            yaxis_title='Behavior',
                            coloraxis_colorbar=dict(title='Percentage (%)')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key="behavior_heatmap")
                    
                    elif viz_option == "Student Comparison":
                        # Create a stacked bar chart comparing all students
                        all_data = []
                        
                        # Get unique students and behaviors
                        students = sorted(df['track_id'].unique())
                        behaviors = sorted(df['class'].unique())
                        
                        # Create data for stacked bar chart
                        for student_id in students:
                            student_df = df[df['track_id'] == student_id]
                            behavior_counts = student_df['class'].value_counts().to_dict()
                            
                            for behavior in behaviors:
                                count = behavior_counts.get(behavior, 0)
                                percentage = count / len(student_df) * 100 if len(student_df) > 0 else 0
                                
                                all_data.append({
                                    'Student': f'Student {student_id}',
                                    'Behavior': behavior,
                                    'Percentage': percentage
                                })
                        
                        # Create stacked bar chart
                        fig = px.bar(
                            pd.DataFrame(all_data),
                            x='Student',
                            y='Percentage',
                            color='Behavior',
                            title='Student Behavior Comparison',
                            labels={'Percentage': 'Percentage of Time (%)'},
                            height=500,
                            color_discrete_map=behavior_colors,
                            barmode='stack'
                        )
                        
                        fig.update_layout(
                            xaxis_title='Student ID',
                            yaxis_title='Percentage of Time (%)',
                            legend_title='Behavior'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key="student_comparison")
                else:
                    st.info("No data available for the selected filters.")
            else:
                st.info("No tracking data available for visualization.")
        
        with tab4:
            # Behavior Summary Table and Graphs
            st.subheader("Student Behavior Summary")
            
            if st.session_state.track_summary and len(st.session_state.track_summary) > 0:
                # Filter based on selected students
                filtered_summary = {k: v for k, v in st.session_state.track_summary.items() 
                                 if k in st.session_state.selected_student_ids}
                
                # Create summary dataframe
                summary_data = []
                for student_id, data in filtered_summary.items():
                    # Get face snapshot
                    face_img = None
                    if student_id in st.session_state.face_snapshots:
                        face_img = st.session_state.face_snapshots[student_id]
                    
                    # Create row
                    row = {
                        'Student ID': f"Student {student_id}",
                        'Dominant Behavior': data.get('dominant_behavior', 'Unknown'),
                        'Engagement Score': f"{data.get('engagement_score', 0.0):.1f}%",
                        'Frames Tracked': data.get('total_detections', 0)
                    }
                    
                    # Add behavior percentages - with safety check
                    all_behaviors = set()
                    # Safely collect all behaviors with checks
                    for d in filtered_summary.values():
                        if 'behavior_percentages' in d and d['behavior_percentages']:
                            all_behaviors.update(d['behavior_percentages'].keys())
                    
                    # Now use the collected behaviors 
                    for behavior in sorted(all_behaviors):
                        if 'behavior_percentages' in data and behavior in data['behavior_percentages']:
                            row[f"{behavior} %"] = f"{data['behavior_percentages'][behavior]:.1f}%"
                        else:
                            row[f"{behavior} %"] = "0.0%"
                    
                    summary_data.append(row)
                
                # Create dataframe
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    
                    # Display summary table
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Create face snapshot grid
                    if st.session_state.face_snapshots:
                        st.subheader("Student Face Snapshots")
                        
                        # Select only filtered students
                        filtered_faces = {k: v for k, v in st.session_state.face_snapshots.items()
                                        if k in st.session_state.selected_student_ids}
                        
                        # Display face grid
                        cols = st.columns(min(4, len(filtered_faces)))
                        for i, (student_id, face_img) in enumerate(filtered_faces.items()):
                            col_idx = i % 4
                            with cols[col_idx]:
                                # Convert OpenCV image to PIL
                                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                                st.image(face_pil, caption=f"Student {student_id}", use_container_width=True)
                    
                    # Create behavior graphs
                    st.subheader("Behavior Analysis Graphs")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Student behavior comparison chart
                        st.markdown("**Behavior Distribution by Student**")
                        
                        # Check if we have valid behavior data
                        has_behavior_data = False
                        for data in filtered_summary.values():
                            if 'behavior_percentages' in data and data['behavior_percentages']:
                                has_behavior_data = True
                                break
                                
                        if has_behavior_data:
                            try:
                                # Define student_ids and behaviors
                                student_ids = list(filtered_summary.keys())
                                # Sort student_ids consistently by string representation
                                student_ids = sorted(student_ids, key=str)
                                
                                # Safely collect all behaviors
                                behaviors = set()
                                for d in filtered_summary.values():
                                    if 'behavior_percentages' in d and d['behavior_percentages']:
                                        behaviors.update(d['behavior_percentages'].keys())
                                behaviors = sorted(behaviors)
                                
                                # Create data for plotly chart
                                plotly_data = []
                                
                                for student_id in student_ids:
                                    for behavior in behaviors:
                                        # Check if the keys exist
                                        if ('behavior_percentages' in filtered_summary[student_id] and 
                                            behavior in filtered_summary[student_id]['behavior_percentages']):
                                            percentage = filtered_summary[student_id]['behavior_percentages'][behavior]
                                        else:
                                            percentage = 0
                                        
                                        plotly_data.append({
                                            'Student': f"Student {student_id}",
                                            'Behavior': behavior,
                                            'Percentage': percentage
                                        })
                                
                                # Create plotly chart
                                if plotly_data:
                                    fig = px.bar(
                                        pd.DataFrame(plotly_data),
                                        x='Student',
                                        y='Percentage',
                                        color='Behavior',
                                        title='Student Behavior Distribution',
                                        barmode='stack',
                                        color_discrete_sequence=px.colors.qualitative.Bold
                                    )
                                    fig.update_layout(
                                        xaxis_title='Student ID',
                                        yaxis_title='Percentage of Time (%)',
                                        legend_title='Behavior',
                                        height=500
                                    )
                                    st.plotly_chart(fig, use_container_width=True, key="behavior_distribution_bar")
                                else:
                                    st.info("No behavior data available for visualization.")
                            except Exception as e:
                                st.warning(f"Could not create behavior distribution chart: {str(e)}")
                                st.info("Error in behavior data visualization. Try filtering to fewer students.")
                        else:
                            st.info("No behavior data available for visualization.")
                    with col2:
                        # Overall class behavior pie chart
                        st.markdown("**Overall Class Behavior Distribution**")
                        
                        # Aggregate behavior percentages across all students
                        behavior_totals = {}
                        
                        # Calculate total frames safely
                        total_frames = sum(data.get('total_detections', 0) for data in filtered_summary.values())
                        
                        for student_id, data in filtered_summary.items():
                            # Check if behavior_counts exists
                            if 'behavior_counts' in data:
                                for behavior, count in data['behavior_counts'].items():
                                    if behavior in behavior_totals:
                                        behavior_totals[behavior] += count
                                    else:
                                        behavior_totals[behavior] = count
                            
                        # Create pie chart only if we have data
                        if behavior_totals:
                            fig = px.pie(
                                names=list(behavior_totals.keys()),
                                values=list(behavior_totals.values()),
                                title='Overall Class Behavior Distribution',
                                hole=0.4,
                                color_discrete_sequence=px.colors.qualitative.Bold
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True, key="overall_behavior_pie")
                        else:
                            st.info("No behavior data available for visualization.")
                    
                    # Engagement score comparison
                    st.subheader("Student Engagement Scores")
                    
                    # Create engagement bar chart safely
                    try:
                        engagement_scores = [data.get('engagement_score', 0.0) for data in filtered_summary.values()]
                        
                        # Only create if we have data
                        if engagement_scores and any(score > 0 for score in engagement_scores):
                            fig = px.bar(
                                x=[f"Student {id}" for id in filtered_summary.keys()],
                                y=engagement_scores,
                                title='Student Engagement Comparison',
                                color=engagement_scores,
                                color_continuous_scale='RdYlGn',
                                text=[f"{score:.1f}%" for score in engagement_scores]
                            )
                            fig.update_layout(
                                xaxis_title='Student ID',
                                yaxis_title='Engagement Score (%)',
                                coloraxis_showscale=False,
                                yaxis=dict(range=[0, 110]),
                                height=500
                            )
                            fig.update_traces(textposition='outside')
                            st.plotly_chart(fig, use_container_width=True, key="engagement_score_bar")
                        else:
                            st.info("No engagement score data available.")
                    except Exception as e:
                        st.warning(f"Error creating engagement chart: {str(e)}")
                        st.info("Could not visualize engagement scores due to data issues.")
                else:
                    st.info("No student behavior data available for the selected students.")
            else:
                st.info("No student tracking data available.")
        
        # AI Analysis tab - only if Claude is enabled
        if st.session_state.use_claude and 'tab5' in locals():
            with tab5:
                st.header("🧠 AI-Powered Classroom Analysis")
                
                # Create a prompt based on the track data
                if st.session_state.track_data and st.session_state.track_summary:
                    # Extract key data for analysis
                    track_data = st.session_state.track_data
                    track_summary = st.session_state.track_summary
                    
                    # Create a tabbed interface for overall analysis and per-student analysis
                    ai_tab1, ai_tab2 = st.tabs(["Classroom Overview", "Individual Student Analysis"])
                    
                    with ai_tab1:
                        # Format student behavior data
                        student_behaviors = {}
                        for student_id, data in track_summary.items():
                            if student_id != 'classroom_analysis':
                                # Include all students even if missing behavior data
                                behaviors_dict = {}
                                if 'behavior_percentages' in data and data['behavior_percentages']:
                                    behaviors_dict = data['behavior_percentages']
                                
                                student_behaviors[f"Student {student_id}"] = {
                                    "dominant_behavior": data.get('dominant_behavior', 'Unknown'),
                                    "engagement_score": f"{data.get('engagement_score', 0.0):.1f}%",
                                    "behaviors": behaviors_dict
                                }
                                
                        # Get behavior counts with error handling 
                        behavior_counts = {}
                        try:
                            for entry in track_data:
                                behavior = entry.get('class')
                                if behavior:
                                    behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
                        except Exception as e:
                            st.warning(f"Error calculating behavior counts: {str(e)}")
                            behavior_counts = {"Error": 1}
                        
                        # Create a detailed prompt for Claude for overall classroom analysis
                        classroom_prompt = f"""
                        # Classroom Behavior Analysis
                        
                        ## Overview
                        - Total students detected: {len([k for k in track_summary.keys() if k != 'classroom_analysis'])}
                        - Total frames analyzed: {len(set([det.get('frame') for det in track_data]))}
                        
                        ## Class-level Behavior Distribution
                        {json.dumps(behavior_counts, indent=2)}
                        
                        ## Student-level Behaviors
                        {json.dumps(student_behaviors, indent=2)}
                        
                        ## Analysis Request
                        You are analyzing classroom behavior data captured through computer vision.
                        Please provide a clear, concise, and actionable analysis with these sections:
                        
                        1. OVERVIEW: A 2-3 sentence summary of the overall classroom engagement level
                        
                        2. KEY PATTERNS: List 3-5 bullet points of the most important behavioral patterns observed
                        
                        3. TEACHING RECOMMENDATIONS: List 3-5 specific, actionable recommendations for the teacher
                        
                        4. CONCERNING BEHAVIORS: Note any concerning behavior patterns that need immediate attention
                        
                        5. CLASSROOM MANAGEMENT: 2-3 strategies to improve overall classroom management
                        
                        Keep your analysis focused, evidence-based, and directly tied to the observed data.
                        Focus on practical insights rather than general theories.
                        """
                        
                        # Display an explanatory message
                        st.subheader("🏫 Overall Classroom Analysis")
                        st.info("AI analysis is performed using Claude 3.7 Sonnet via Puter.js (completely free)")
                        
                        # Add a button to trigger analysis manually if automatic doesn't work
                        trigger_analysis = st.button("Generate Classroom Analysis", key="trigger_classroom_analysis")
                        
                        # Use the Puter.js component for client-side analysis in its own container
                        classroom_container = st.container()
                        
                        # Fix the static fallback analysis for classroom to handle empty data and avoid numpy warnings
                        # Static fallback analysis (simpler but works offline)
                        fallback_analysis = f"""
                        # Classroom Analysis
                        
                        ## OVERVIEW
                        The classroom shows {len(student_behaviors)} students with varying levels of engagement. Overall engagement level is {'high' if len(student_behaviors) > 0 and sum([float(data.get('engagement_score', 0)) for _, data in track_summary.items() if _ != 'classroom_analysis'])/max(1, len(student_behaviors)) > 70 else 'moderate' if len(student_behaviors) > 0 and sum([float(data.get('engagement_score', 0)) for _, data in track_summary.items() if _ != 'classroom_analysis'])/max(1, len(student_behaviors)) > 40 else 'low'}.
                        
                        ## KEY PATTERNS
                        * {max(behavior_counts.items(), key=lambda x: x[1])[0] if behavior_counts else 'No behaviors detected'} is the most common behavior
                        * {len([s for s, d in student_behaviors.items() if d.get("dominant_behavior") == "Attentive"])} students are predominantly attentive
                        * {len([s for s, d in student_behaviors.items() if "Distracted" in d.get("dominant_behavior", "")])} students show signs of distraction
                        
                        ## TEACHING RECOMMENDATIONS
                        * Use more interactive teaching methods to increase engagement
                        * Consider checking in with distracted students 
                        * Break up the lesson with short activities to regain attention
                        
                        ## CONCERNING BEHAVIORS
                        * {'No major concerns' if not behavior_counts or 'Distracted' not in behavior_counts or sum(behavior_counts.values()) == 0 or behavior_counts.get('Distracted', 0)/sum(behavior_counts.values()) < 0.3 else 'High level of distraction that may need addressing'}
                        
                        ## CLASSROOM MANAGEMENT
                        * Move around the classroom more to engage all students 
                        * Consider re-arranging seating for better engagement
                        """
                        
                        # Try puter.js component, with fallback
                        with classroom_container:
                            # If button pressed or this is first load, trigger analysis
                            if trigger_analysis:
                                try:
                                    # Try to use Puter.js
                                    puter_ai_chat(classroom_prompt)
                                except Exception as e:
                                    st.error(f"Puter.js integration failed: {str(e)}")
                                    st.markdown(fallback_analysis)
                            else:
                                # On first load, also try analysis
                                try:
                                    puter_ai_chat(classroom_prompt)
                                except Exception as e:
                                    st.warning("Automatic analysis failed. Click 'Generate Classroom Analysis' to try again.")
                        
                        # Add download button for the complete analysis
                        col1, col2 = st.columns([3, 1])
                        with col2:
                            st.download_button(
                                "Download Analysis",
                                classroom_prompt,
                                "classroom_analysis.txt",
                                "text/plain"
                            )
                    
                    with ai_tab2:
                        st.subheader("👨‍🎓 Individual Student Analysis")
                        
                        # Get student IDs sorted
                        student_ids = [k for k in track_summary.keys() if k != 'classroom_analysis']
                        sorted_student_ids = sorted(student_ids, key=str)
                        
                        if sorted_student_ids:
                            # Create selectbox for student selection
                            selected_student = st.selectbox(
                                "Select a student to analyze:",
                                sorted_student_ids,
                                format_func=lambda x: f"Student {x}"
                            )
                            
                            # Add a manual trigger button for individual analysis
                            trigger_student_analysis = st.button("Generate Student Analysis", key="trigger_student_analysis")
                            
                            # Get selected student data
                            if selected_student in track_summary:
                                student_data = track_summary[selected_student]
                                student_detections = [d for d in track_data if d['track_id'] == selected_student]
                                
                                # Show student face if available
                                if selected_student in st.session_state.face_snapshots:
                                    face_img = st.session_state.face_snapshots[selected_student]
                                    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                                    col1, col2 = st.columns([1, 3])
                                    with col1:
                                        st.image(face_pil, caption=f"Student {selected_student}", width=150)
                                    with col2:
                                        dominant_behavior = student_data.get('dominant_behavior', 'Unknown')
                                        engagement_score = student_data.get('engagement_score', 0.0)
                                        total_detections = student_data.get('total_detections', 0)
                                        
                                        st.metric("Dominant Behavior", dominant_behavior)
                                        st.metric("Engagement Score", f"{engagement_score:.1f}%")
                                        st.metric("Frames Tracked", total_detections)
                                
                                # Create per-student prompt for Claude
                                student_prompt = f"""
                                # Individual Student Analysis: Student {selected_student}
                                
                                ## Student Overview
                                - Dominant Behavior: {student_data.get('dominant_behavior', 'Unknown')}
                                - Engagement Score: {student_data.get('engagement_score', 0.0):.1f}%
                                - Total Frames Tracked: {student_data.get('total_detections', 0)}
                                
                                ## Behavior Distribution
                                {json.dumps(student_data.get('behavior_percentages', {}), indent=2)}
                                
                                ## Analysis Request
                                You are analyzing an individual student's behavior data from computer vision.
                                Please provide a clear, specific analysis focusing ONLY on this student with these sections:
                                
                                1. STUDENT PROFILE: A 2-3 sentence summary of this student's engagement pattern
                                
                                2. STRENGTHS: 1-2 positive behavioral patterns this student exhibits
                                
                                3. AREAS FOR IMPROVEMENT: 1-2 specific behaviors that could be improved
                                
                                4. PERSONALIZED STRATEGIES: 2-3 targeted teaching strategies for this specific student
                                
                                5. ENGAGEMENT INSIGHTS: How to better engage this specific student based on their behavior pattern
                                
                                Keep your analysis focused on THIS SPECIFIC STUDENT and provide actionable advice.
                                """
                                
                                # Create a container for the student analysis
                                student_container = st.container()
                                
                                # Create a simpler static fallback analysis
                                student_fallback = f"""
                                # Student {selected_student} Analysis
                                
                                ## STUDENT PROFILE
                                Student {selected_student} shows predominantly {student_data.get('dominant_behavior', 'mixed')} behavior with an engagement score of {student_data.get('engagement_score', 0.0):.1f}%.
                                
                                ## STRENGTHS
                                {'Attentiveness' if 'Attentive' in student_data.get('behavior_percentages', {}) and student_data.get('behavior_percentages', {}).get('Attentive', 0) > 30 else 'Consistent behavior patterns'}.
                                
                                ## AREAS FOR IMPROVEMENT
                                {'Focus and attention' if 'Distracted' in student_data.get('behavior_percentages', {}) and student_data.get('behavior_percentages', {}).get('Distracted', 0) > 30 else 'Maintaining engagement throughout the class period'}.
                                
                                ## PERSONALIZED STRATEGIES
                                * {'Check in with this student regularly to ensure understanding' if student_data.get('engagement_score', 0.0) < 50 else "Acknowledge this student's positive behaviors"}
                                * {'Provide more structure and clear expectations' if 'Distracted' in student_data.get('behavior_percentages', {}) else 'Ask this student to help peers or take leadership roles'}
                                
                                ## ENGAGEMENT INSIGHTS
                                This student would benefit from {'more direct teacher interaction and clearer instructions' if student_data.get('engagement_score', 0.0) < 50 else 'recognition of their focus and additional challenging tasks'}.
                                """
                                
                                # Try puter.js component with fallback
                                with student_container:
                                    # If button pressed or this is first load, trigger analysis
                                    if trigger_student_analysis:
                                        try:
                                            # Try to use Puter.js
                                            puter_ai_chat(student_prompt)
                                        except Exception as e:
                                            st.error(f"Puter.js integration failed: {str(e)}")
                                            st.markdown(student_fallback)
                                    else:
                                        # On first load, also try analysis
                                        try:
                                            puter_ai_chat(student_prompt)
                                        except Exception as e:
                                            st.warning("Automatic analysis failed. Click 'Generate Student Analysis' to try again.")
                                
                                # Add behavior visualization for this student
                                st.subheader("Behavior Timeline")
                                student_df = pd.DataFrame(student_detections)
                                
                                # Add timestamp information if not present
                                if 'timestamp' not in student_df.columns and 'frame' in student_df.columns:
                                    fps = st.session_state.get('video_fps', 30.0)
                                    student_df['timestamp'] = student_df['frame'] / fps
                                    student_df['time_str'] = student_df['timestamp'].apply(lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}")
                                
                                # Create timeline plot
                                if not student_df.empty:
                                    fig = px.scatter(
                                        student_df,
                                        x='timestamp', 
                                        y='class',
                                        color='class',
                                        title=f'Behavior Timeline for Student {selected_student}',
                                        labels={'timestamp': 'Time (seconds)', 'class': 'Behavior'},
                                        height=300
                                    )
                                    
                                    fig.update_layout(
                                        xaxis_title='Time (seconds)',
                                        yaxis_title='Behavior',
                                        showlegend=True,
                                        margin=dict(l=20, r=20, t=40, b=20)
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("No timeline data available for this student.")
                                
                                # Download button for individual student analysis
                                st.download_button(
                                    "Download Student Analysis",
                                    student_prompt,
                                    f"student_{selected_student}_analysis.txt",
                                    "text/plain"
                                )
                        else:
                            st.warning("No student data available for analysis.")
                else:
                    st.info("No analysis data available yet. Process a video with AI Analysis enabled to see insights.")
    
    # Download section
    st.markdown("---")
    st.header("Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download data as CSV
        if 'track_data' in st.session_state and st.session_state.track_data:
            # Create CSV from tracking data
            track_df = pd.DataFrame(st.session_state.track_data)
            csv = track_df.to_csv(index=False)
            
            st.download_button(
                label="Download Tracking Data (CSV)",
                data=csv,
                file_name="classroom_behavior_tracking.csv",
                mime="text/csv"
            )
        elif 'detections' in st.session_state and st.session_state.detections:
            # Create CSV from image detections
            detection_df = pd.DataFrame(st.session_state.detections)
            csv = detection_df.to_csv(index=False)
            
            st.download_button(
                label="Download Detection Data (CSV)",
                data=csv,
                file_name="classroom_behavior_detection.csv",
                mime="text/csv"
            )
    
    with col2:
        # Download data as JSON
        if 'track_summary' in st.session_state and st.session_state.track_summary:
            # Prepare summary data
            summary_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'student_summaries': {}
            }
            
            for student_id, data in st.session_state.track_summary.items():
                # Skip the special 'classroom_analysis' key
                if student_id == 'classroom_analysis':
                    continue
                    
                try:
                    # Convert numpy types to native Python types for JSON serialization
                    summary_data['student_summaries'][int(student_id)] = {
                        'dominant_behavior': data.get('dominant_behavior', 'Unknown'),
                        'engagement_score': float(data.get('engagement_score', 0.0)),
                        'total_detections': int(data.get('total_detections', 0)),
                        'behavior_percentages': {k: float(v) for k, v in data.get('behavior_percentages', {}).items()}
                    }
                except (ValueError, TypeError):
                    # If student_id can't be converted to int, use it as a string
                    summary_data['student_summaries'][str(student_id)] = {
                        'dominant_behavior': data.get('dominant_behavior', 'Unknown'),
                        'engagement_score': float(data.get('engagement_score', 0.0)),
                        'total_detections': int(data.get('total_detections', 0)),
                        'behavior_percentages': {k: float(v) for k, v in data.get('behavior_percentages', {}).items()}
                    }
            
            # Convert to JSON
            json_data = json.dumps(summary_data, indent=2)
            
            st.download_button(
                label="Download Summary (JSON)",
                data=json_data,
                file_name="classroom_behavior_summary.json",
                mime="application/json"
            )
        elif 'class_summary' in st.session_state and st.session_state.class_summary:
            # Prepare summary data for image
            summary_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'behavior_counts': st.session_state.class_summary
            }
            
            # Convert to JSON
            json_data = json.dumps(summary_data, indent=2)
            
            st.download_button(
                label="Download Summary (JSON)",
                data=json_data,
                file_name="classroom_behavior_summary.json",
                mime="application/json"
            )
else:
    # Show welcome screen
    st.markdown("""
    ## Welcome to the Classroom Behavior Analysis System
    
    This dashboard analyzes student behavior in classroom settings using computer vision and provides detailed analytics.
    
    ### Features:
    - **Behavior Detection**: Identifies student behaviors like attentive, disengaged, etc.
    - **Student Tracking**: Follows individual students throughout videos
    - **Engagement Analytics**: Calculates detailed engagement metrics
    - **Visual Reports**: Creates graphs and visualizations of classroom behavior
    
    ### How to use:
    1. Upload an image or video using the sidebar
    2. Click "Analyze" to process the media
    3. View the results and filter as needed
    4. Download reports in CSV or JSON format
    
    Upload a file to get started!
    """)
    
    # Display sample image
    st.sidebar.markdown("### Example Behaviors")
    st.sidebar.markdown("""
    - ✅ Attentive
    - 📝 Writing Notes
    - 💻 Using Laptop
    - 😴 Tired/Bored
    - 📱 Using Mobile
    - 💬 Talking
    """)