"""
Configuration settings for the Classroom Behavior Analysis System
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "runs/train/yolov8_classroom/weights/best.pt")
DATA_YAML_PATH = os.path.join(ROOT_DIR, "dataset/data.yaml")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Class mapping (combining tired and bored into disengaged)
CLASS_MAPPING = {
    0: 'attentive',    # Attentive -> attentive
    1: 'disengaged',   # Boord -> disengaged
    2: 'distracted',   # Distractive -> distracted
    3: 'disengaged',   # Eyes Closed -> disengaged
    4: 'disengaged',   # Face Covered -> disengaged
    5: 'disengaged',   # Facing Down -> disengaged
    6: 'distracted',   # Laughing -> distracted
    7: 'attentive',    # Normal activity -> attentive
    8: 'unknown',      # Outside Classroom -> unknown
    9: 'distracted',   # Seeing Outside -> distracted
    10: 'disengaged',  # Sleeping -> disengaged
    11: 'distracted',  # Standing -> distracted
    12: 'distracted',  # Talking -> distracted
    13: 'disengaged',  # Tired -> disengaged
    14: 'distracted',  # Using Laptop -> distracted
    15: 'distracted',  # Using Mobile -> distracted
    16: 'attentive',   # Writing Notes -> attentive
    17: 'disengaged'   # facing back -> disengaged
}

# Tracker settings
MAX_AGE = 30
MIN_HITS = 3
TRACKER_IOU_THRESHOLD = 0.3

# Video processing
FRAME_SKIP = 2  # Process every Nth frame for speed

# Training hyperparameters (optimized for better accuracy)
EPOCHS = 300     # Increased epochs
BATCH_SIZE = 16
IMAGE_SIZE = 640
LEARNING_RATE = 0.001  # Adjusted learning rate
PATIENCE = 50    # Increased patience for early stopping 