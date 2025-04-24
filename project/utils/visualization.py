"""
Visualization utilities for Classroom Behavior Analysis System
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def draw_boxes(frame, tracks, conf_threshold=0.5):
    """
    Draw bounding boxes with labels on frame
    
    Args:
        frame: Input frame
        tracks: Tracking information
        conf_threshold: Confidence threshold for displaying boxes
        
    Returns:
        Annotated frame
    """
    # Make a copy of the frame
    annotated_frame = frame.copy()
    
    # Color mapping for behaviors
    color_map = {
        'attentive': (0, 255, 0),    # Green
        'disengaged': (0, 0, 255),   # Red
        'other_behavior': (255, 0, 0), # Blue
        'unknown': (128, 128, 128)   # Gray
    }
    
    # Draw each tracked object
    for track in tracks:
        # Extract information
        track_id = track['track_id']
        bbox = track['bbox']
        behavior = track['behavior']
        confidence = track.get('confidence', 0)
        
        # Skip low confidence detections
        if confidence < conf_threshold:
            continue
        
        # Get color for behavior
        color = color_map.get(behavior, (128, 128, 128))
        
        # Draw bounding box
        x1, y1, x2, y2 = [int(c) for c in bbox]
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"ID:{track_id} {behavior} {confidence:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated_frame, (x1, y1-25), (x1+label_width, y1), color, -1)
        
        # Draw label
        cv2.putText(annotated_frame, label, (x1, y1-7), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return annotated_frame

def create_attention_heatmap(frame, tracks, alpha=0.6):
    """
    Create an attention heatmap based on student behaviors
    
    Args:
        frame: Input frame
        tracks: Tracking information
        alpha: Transparency of heatmap overlay
        
    Returns:
        Frame with heatmap overlay
    """
    # Create empty heatmap
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    
    # Add heat for each person based on behavior
    for track in tracks:
        bbox = track['bbox']
        behavior = track['behavior']
        confidence = track.get('confidence', 0)
        
        # Skip low confidence detections
        if confidence < 0.5:
            continue
        
        # Get heat value based on behavior
        heat_value = 0.0
        if behavior == 'attentive':
            heat_value = 1.0
        elif behavior == 'disengaged':
            heat_value = 0.3
        else:
            heat_value = 0.5
        
        # Apply Gaussian heat to the region
        x1, y1, x2, y2 = [int(c) for c in bbox]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        sigma = max(x2-x1, y2-y1) // 3
        
        # Create a gaussian mask
        y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
        mask = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        
        # Add to heatmap
        heatmap += mask * heat_value * confidence
    
    # Normalize heatmap
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Create colormap
    cmap = LinearSegmentedColormap.from_list('attention_cmap', 
                                             [(0, 'blue'), (0.5, 'yellow'), (1, 'red')])
    
    # Convert heatmap to RGB
    heatmap_rgb = plt.cm.get_cmap(cmap)(heatmap)
    heatmap_rgb = (heatmap_rgb[:, :, :3] * 255).astype(np.uint8)
    
    # Resize heatmap to match frame
    heatmap_rgb = cv2.resize(heatmap_rgb, (frame.shape[1], frame.shape[0]))
    
    # Blend heatmap with original frame
    blended = cv2.addWeighted(frame, 1-alpha, heatmap_rgb, alpha, 0)
    
    return blended

def draw_engagement_gauge(frame, engagement_score, position=(50, 50), size=100):
    """
    Draw an engagement gauge on the frame
    
    Args:
        frame: Input frame
        engagement_score: Engagement score (0-100)
        position: Position of the gauge (x, y)
        size: Size of the gauge
        
    Returns:
        Frame with gauge
    """
    # Make a copy of the frame
    annotated_frame = frame.copy()
    
    # Ensure engagement score is in range 0-100
    engagement_score = max(0, min(100, engagement_score))
    
    # Calculate gauge parameters
    center_x, center_y = position
    radius = size // 2
    start_angle = 180
    end_angle = 0
    angle = start_angle - (engagement_score / 100) * (start_angle - end_angle)
    
    # Draw gauge background
    cv2.ellipse(annotated_frame, (center_x, center_y), (radius, radius), 
                0, start_angle, end_angle, (200, 200, 200), -1)
    
    # Draw gauge value
    cv2.ellipse(annotated_frame, (center_x, center_y), (radius, radius), 
                0, angle, end_angle, (0, 255, 0), -1)
    
    # Draw gauge border
    cv2.ellipse(annotated_frame, (center_x, center_y), (radius, radius), 
                0, start_angle, end_angle, (0, 0, 0), 2)
    
    # Draw gauge text
    cv2.putText(annotated_frame, f"Engagement: {engagement_score:.1f}%", 
                (center_x - radius, center_y + radius + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return annotated_frame

def draw_behavior_summary(frame, behavior_counts, position=(50, 200), size=(300, 150)):
    """
    Draw a behavior summary on the frame
    
    Args:
        frame: Input frame
        behavior_counts: Dictionary of behavior counts
        position: Position of the summary (x, y)
        size: Size of the summary (width, height)
        
    Returns:
        Frame with summary
    """
    # Make a copy of the frame
    annotated_frame = frame.copy()
    
    # Calculate summary parameters
    x, y = position
    width, height = size
    
    # Draw summary background
    cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (240, 240, 240), -1)
    cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (0, 0, 0), 2)
    
    # Draw summary title
    cv2.putText(annotated_frame, "Behavior Summary", (x + 10, y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Draw behavior counts
    y_offset = 60
    for behavior, count in behavior_counts.items():
        # Color for behavior
        if behavior == 'attentive':
            color = (0, 150, 0)
        elif behavior == 'disengaged':
            color = (0, 0, 200)
        else:
            color = (100, 100, 100)
        
        # Draw behavior text
        cv2.putText(annotated_frame, f"{behavior}: {count}", (x + 20, y + y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 30
    
    return annotated_frame 