"""
Behavior logger for tracking student behaviors and saving face snapshots
"""

import os
import json
import cv2
import time
from datetime import datetime

class BehaviorLogger:
    def __init__(self, output_dir="output"):
        """
        Initialize the behavior logger
        
        Args:
            output_dir: Directory to save logs and snapshots
        """
        self.output_dir = output_dir
        
        # Create output directories
        self.log_dir = os.path.join(output_dir, "logs")
        self.snapshot_dir = os.path.join(output_dir, "snapshots")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Initialize log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"behavior_log_{timestamp}.json")
        
        # Initialize behavior history
        self.behavior_history = {}
        
        # Create initial log file with metadata
        self._initialize_log_file()
        
        print(f"Behavior logger initialized. Log file: {self.log_file}")
    
    def _initialize_log_file(self):
        """Initialize the log file with metadata"""
        log_data = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.0",
                "session_id": int(time.time())
            },
            "entries": []
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def log_behaviors(self, frame, tracks, frame_idx, fps):
        """
        Log behavior data and save face snapshots
        
        Args:
            frame: Current video frame
            tracks: List of tracked objects with behavior info
            frame_idx: Current frame index
            fps: Frames per second of the video
        """
        timestamp = frame_idx / fps  # Convert frame index to seconds
        
        # Load existing log data
        with open(self.log_file, 'r') as f:
            log_data = json.load(f)
        
        # Process each track
        for track in tracks:
            track_id = track['track_id']
            behavior = track['behavior']
            confidence = track['confidence']
            bbox = track['bbox']
            
            # Save face snapshot every 30 frames or when behavior changes
            save_snapshot = False
            if track_id not in self.behavior_history:
                save_snapshot = True
                self.behavior_history[track_id] = {"behavior": behavior, "last_snapshot": frame_idx}
            elif self.behavior_history[track_id]["behavior"] != behavior:
                save_snapshot = True
                self.behavior_history[track_id]["behavior"] = behavior
                self.behavior_history[track_id]["last_snapshot"] = frame_idx
            elif frame_idx - self.behavior_history[track_id]["last_snapshot"] >= 30:
                save_snapshot = True
                self.behavior_history[track_id]["last_snapshot"] = frame_idx
            
            # Save face snapshot if needed
            snapshot_path = None
            if save_snapshot:
                snapshot_path = self._save_snapshot(frame, bbox, track_id, frame_idx)
            
            # Create log entry
            log_entry = {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "track_id": track_id,
                "behavior": behavior,
                "confidence": confidence,
                "bbox": bbox,
                "snapshot_path": snapshot_path
            }
            
            # Add to log data
            log_data["entries"].append(log_entry)
        
        # Save updated log data
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def _save_snapshot(self, frame, bbox, track_id, frame_idx):
        """
        Save a snapshot of the face
        
        Args:
            frame: Current video frame
            bbox: Bounding box [x1, y1, x2, y2]
            track_id: Track ID
            frame_idx: Current frame index
        
        Returns:
            snapshot_path: Path to the saved snapshot
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure bbox is within frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        # Extract face region
        face_img = frame[y1:y2, x1:x2]
        
        # Skip if face region is empty
        if face_img.size == 0:
            return None
        
        # Create snapshot filename
        filename = f"student_{track_id}_frame_{frame_idx}.jpg"
        snapshot_path = os.path.join(self.snapshot_dir, filename)
        
        # Save snapshot
        cv2.imwrite(snapshot_path, face_img)
        
        # Return relative path
        return os.path.join("snapshots", filename) 