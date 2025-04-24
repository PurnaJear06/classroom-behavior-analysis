"""
Data processing utilities for Classroom Behavior Analysis System
"""

import os
import cv2
import numpy as np
import json
from collections import defaultdict

def extract_video_frames(video_path, output_dir, frame_interval=30):
    """
    Extract frames from a video at specified intervals
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame
        
    Returns:
        List of paths to extracted frames
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_count = 0
    extracted_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract every Nth frame
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_path)
        
        frame_count += 1
    
    # Release resources
    cap.release()
    
    print(f"Extracted {len(extracted_frames)} frames from {video_path}")
    return extracted_frames

def analyze_behavior_trends(log_file):
    """
    Analyze behavior trends from log file
    
    Args:
        log_file: Path to behavior log file
        
    Returns:
        Dictionary with behavior trend analysis
    """
    # Load log data
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    entries = log_data.get("entries", [])
    if not entries:
        return {"error": "No entries found in log file"}
    
    # Sort entries by timestamp
    entries.sort(key=lambda x: x.get("timestamp", 0))
    
    # Get time range
    start_time = entries[0].get("timestamp", 0)
    end_time = entries[-1].get("timestamp", 0)
    duration = end_time - start_time
    
    # Create time bins (10 bins across the session)
    num_bins = 10
    bin_duration = duration / num_bins
    time_bins = [start_time + i * bin_duration for i in range(num_bins + 1)]
    
    # Initialize data structures
    behavior_bins = defaultdict(lambda: [0] * num_bins)
    student_counts = [0] * num_bins
    
    # Process entries
    for entry in entries:
        timestamp = entry.get("timestamp", 0)
        behavior = entry.get("behavior", "unknown")
        
        # Find bin index
        bin_idx = min(int((timestamp - start_time) / bin_duration), num_bins - 1)
        
        # Count behaviors in bins
        behavior_bins[behavior][bin_idx] += 1
        student_counts[bin_idx] += 1
    
    # Calculate percentages
    behavior_percentages = {}
    for behavior, counts in behavior_bins.items():
        percentages = []
        for i, count in enumerate(counts):
            if student_counts[i] > 0:
                percentages.append(count / student_counts[i] * 100)
            else:
                percentages.append(0)
        behavior_percentages[behavior] = percentages
    
    # Create time labels (in minutes)
    time_labels = [f"{(t - start_time) / 60:.1f}" for t in time_bins]
    
    # Find engagement patterns
    attentive_percentages = behavior_percentages.get("attentive", [0] * num_bins)
    disengaged_percentages = behavior_percentages.get("disengaged", [0] * num_bins)
    
    # Find times of lowest/highest engagement
    if attentive_percentages:
        max_attentive_idx = np.argmax(attentive_percentages)
        min_attentive_idx = np.argmin(attentive_percentages)
        max_attentive_time = float(time_labels[max_attentive_idx])
        min_attentive_time = float(time_labels[min_attentive_idx])
    else:
        max_attentive_time = min_attentive_time = 0
    
    # Create analysis results
    analysis = {
        "session_duration_minutes": duration / 60,
        "time_bins_minutes": time_labels,
        "behavior_percentages": behavior_percentages,
        "peak_engagement_time": max_attentive_time,
        "lowest_engagement_time": min_attentive_time,
        "engagement_trend": "improving" if attentive_percentages[-1] > attentive_percentages[0] else "declining",
        "student_counts": student_counts
    }
    
    return analysis

def calculate_confusion_matrix(detections, ground_truth, iou_threshold=0.5):
    """
    Calculate confusion matrix for behavior detections
    
    Args:
        detections: List of detection dictionaries
        ground_truth: List of ground truth dictionaries
        iou_threshold: IOU threshold for matching
        
    Returns:
        Dictionary with confusion matrix and metrics
    """
    # Initialize confusion matrix
    behaviors = ["attentive", "disengaged", "other_behavior"]
    confusion_matrix = np.zeros((len(behaviors), len(behaviors)), dtype=int)
    
    # Count true positives, false positives, and false negatives
    tp = 0
    fp = 0
    fn = 0
    
    # Match detections to ground truth
    for gt in ground_truth:
        gt_bbox = gt.get("bbox", [0, 0, 0, 0])
        gt_behavior = gt.get("behavior", "unknown")
        gt_behavior_idx = behaviors.index(gt_behavior) if gt_behavior in behaviors else -1
        
        if gt_behavior_idx == -1:
            continue
        
        # Find best matching detection
        best_match = None
        best_iou = 0
        
        for det in detections:
            det_bbox = det.get("bbox", [0, 0, 0, 0])
            iou = calculate_iou(gt_bbox, det_bbox)
            
            if iou > iou_threshold and iou > best_iou:
                best_match = det
                best_iou = iou
        
        if best_match:
            # Found a matching detection
            det_behavior = best_match.get("behavior", "unknown")
            det_behavior_idx = behaviors.index(det_behavior) if det_behavior in behaviors else -1
            
            if det_behavior_idx != -1:
                # Update confusion matrix
                confusion_matrix[gt_behavior_idx, det_behavior_idx] += 1
                
                # Count as true positive if behavior matches
                if gt_behavior_idx == det_behavior_idx:
                    tp += 1
                else:
                    fp += 1
        else:
            # No matching detection, count as false negative
            fn += 1
    
    # Count additional false positives (unmatched detections)
    for det in detections:
        det_bbox = det.get("bbox", [0, 0, 0, 0])
        det_behavior = det.get("behavior", "unknown")
        det_behavior_idx = behaviors.index(det_behavior) if det_behavior in behaviors else -1
        
        if det_behavior_idx == -1:
            continue
        
        # Check if detection matches any ground truth
        matched = False
        for gt in ground_truth:
            gt_bbox = gt.get("bbox", [0, 0, 0, 0])
            iou = calculate_iou(gt_bbox, det_bbox)
            
            if iou > iou_threshold:
                matched = True
                break
        
        if not matched:
            # No matching ground truth, count as false positive
            fp += 1
    
    # Calculate metrics
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    # Create results
    results = {
        "confusion_matrix": confusion_matrix.tolist(),
        "behaviors": behaviors,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
    
    return results

def calculate_iou(bbox1, bbox2):
    """
    Calculate IOU between two bounding boxes
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IOU value
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