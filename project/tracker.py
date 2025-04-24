"""
DeepSORT tracker implementation for student tracking
"""

import numpy as np
from typing import List, Dict
import config

class DeepSORTTracker:
    def __init__(self, max_age=config.MAX_AGE, min_hits=config.MIN_HITS, 
                 iou_threshold=config.TRACKER_IOU_THRESHOLD):
        """
        Initialize DeepSORT tracker
        
        Args:
            max_age: Maximum number of frames to keep track of disappeared object
            min_hits: Minimum number of hits to start tracking
            iou_threshold: IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks = []  # List of active tracks
        
        # Import deep_sort modules here if available
        try:
            # Note: For full DeepSORT implementation, you would import:
            # from deep_sort.deep_sort import nn_matching
            # from deep_sort.deep_sort.tracker import Tracker
            # from deep_sort.deep_sort.detection import Detection
            # from deep_sort.tools import generate_detections
            pass
        except ImportError:
            print("Warning: deep_sort modules not found. Using simplified tracker.")
    
    def update(self, detections: List[Dict], frame) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dictionaries with 'bbox', 'confidence', etc.
            frame: Current video frame
        
        Returns:
            tracks: List of dictionaries with tracking info
        """
        # Placeholder for DeepSORT implementation
        # In a complete implementation, this would:
        # 1. Convert detections to DeepSORT detection objects
        # 2. Extract appearance features for each detection
        # 3. Update the tracker with new detections
        # 4. Return updated tracks
        
        # Simplified implementation (IOU-based tracking)
        updated_tracks = []
        
        # If there are no tracks yet, initialize from detections
        if not self.tracks:
            for det in detections:
                # Create new track
                track = {
                    'track_id': self.next_id,
                    'bbox': det['bbox'],
                    'behavior': det['behavior'],
                    'confidence': det['confidence'],
                    'age': 1,
                    'hits': 1,
                    'time_since_update': 0
                }
                self.tracks.append(track)
                updated_tracks.append(track)
                self.next_id += 1
            return updated_tracks
        
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections_to_tracks(detections)
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx]['bbox'] = detections[det_idx]['bbox']
            self.tracks[track_idx]['behavior'] = detections[det_idx]['behavior']
            self.tracks[track_idx]['confidence'] = detections[det_idx]['confidence']
            self.tracks[track_idx]['hits'] += 1
            self.tracks[track_idx]['time_since_update'] = 0
            self.tracks[track_idx]['age'] += 1
            
            # Add to result if confirmed
            if self.tracks[track_idx]['hits'] >= self.min_hits:
                updated_tracks.append(self.tracks[track_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            track = {
                'track_id': self.next_id,
                'bbox': detections[det_idx]['bbox'],
                'behavior': detections[det_idx]['behavior'],
                'confidence': detections[det_idx]['confidence'],
                'age': 1,
                'hits': 1,
                'time_since_update': 0
            }
            self.tracks.append(track)
            self.next_id += 1
        
        # Update unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx]['time_since_update'] += 1
            self.tracks[track_idx]['age'] += 1
            
            # Add to result if still active
            if self.tracks[track_idx]['time_since_update'] <= self.max_age:
                if self.tracks[track_idx]['hits'] >= self.min_hits:
                    updated_tracks.append(self.tracks[track_idx])
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
        
        return updated_tracks
    
    def _match_detections_to_tracks(self, detections):
        """
        Match detections to existing tracks using IOU
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        # Calculate IOU between each detection and track
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t_idx, track in enumerate(self.tracks):
            for d_idx, detection in enumerate(detections):
                iou_matrix[t_idx, d_idx] = self._calculate_iou(track['bbox'], detection['bbox'])
        
        # Apply Hungarian algorithm here in full implementation
        # For simplicity, use greedy matching
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        # Find matches
        for t_idx in range(len(self.tracks)):
            for d_idx in range(len(detections)):
                if d_idx in unmatched_detections and t_idx in unmatched_tracks:
                    if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                        matched_indices.append((t_idx, d_idx))
                        unmatched_detections.remove(d_idx)
                        unmatched_tracks.remove(t_idx)
                        break
        
        return matched_indices, unmatched_detections, unmatched_tracks
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate IOU between two bounding boxes
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate area of each box
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate IOU
        union_area = area1 + area2 - intersection_area
        iou = intersection_area / union_area
        
        return iou 