"""
Main script for running inference on videos with YOLOv8 and DeepSORT tracking
"""

import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from tracker import DeepSORTTracker
from logger import BehaviorLogger
from summary_generator import SummaryGenerator
import config
from google.colab import drive

class BehaviorAnalysisSystem:
    def __init__(self, model_path, video_path, output_dir=config.OUTPUT_DIR):
        """
        Initialize Behavior Analysis System
        
        Args:
            model_path: Path to YOLOv8 model weights
            video_path: Path to input video
            output_dir: Directory to save outputs
        """
        print(f"Initializing Behavior Analysis System...")
        print(f"Model: {model_path}")
        print(f"Video: {video_path}")
        print(f"Output: {output_dir}")
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSORTTracker(
            max_age=config.MAX_AGE,
            min_hits=config.MIN_HITS,
            iou_threshold=config.TRACKER_IOU_THRESHOLD
        )
        
        # Initialize behavior logger
        self.logger = BehaviorLogger(output_dir=output_dir)
        
        # Class mapping
        self.class_mapping = config.CLASS_MAPPING
        
    def process_frame(self, frame, frame_idx):
        """
        Process a single frame
        
        Args:
            frame: Input frame
            frame_idx: Frame index
            
        Returns:
            Annotated frame
        """
        # Run YOLOv8 inference
        results = self.model(frame)
        
        # Extract detection information
        detections = []
        for r in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls_id = r
            
            # Skip low confidence detections
            if conf < config.CONFIDENCE_THRESHOLD:
                continue
                
            # Map to combined behavior classes
            behavior = self.class_mapping.get(int(cls_id), 'unknown')
            
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'class_id': int(cls_id),
                'behavior': behavior
            })
        
        # Update tracker with new detections
        tracks = self.tracker.update(detections, frame)
        
        # Log behavior data and save face snapshots
        self.logger.log_behaviors(frame, tracks, frame_idx, self.fps)
        
        # Draw bounding boxes and info on frame
        annotated_frame = self.draw_results(frame, tracks)
        
        return annotated_frame
    
    def draw_results(self, frame, tracks):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            tracks: Tracking information
            
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
            confidence = track['confidence']
            
            # Get color for behavior
            color = color_map.get(behavior, (128, 128, 128))
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"ID:{track_id} {behavior} {confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1-25), (x1+label_width, y1), color, -1)
            
            # Draw label
            cv2.putText(annotated_frame, label, (x1, y1-7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def process_video(self):
        """
        Process the entire video
        """
        frame_idx = 0
        processed_count = 0
        
        # Create video writer
        output_path = os.path.join(self.output_dir, "annotated_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        print(f"Processing video with {self.total_frames} frames...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Process every Nth frame for speed
            if frame_idx % config.FRAME_SKIP == 0:
                # Process current frame
                annotated_frame = self.process_frame(frame, frame_idx)
                
                # Write frame to output video
                out.write(annotated_frame)
                
                processed_count += 1
                
                # Display progress every 100 frames
                if processed_count % 100 == 0:
                    progress = (frame_idx / self.total_frames) * 100
                    print(f"Processed {processed_count} frames ({progress:.1f}%)")
            
            # Increment frame index
            frame_idx += 1
        
        # Release resources
        self.cap.release()
        out.release()
        
        print(f"Video processing complete. Processed {processed_count} frames.")
        print(f"Output video saved to {output_path}")
        
        # Generate summary
        print("Generating engagement summary...")
        summary_generator = SummaryGenerator(self.logger.log_file)
        summary_file = summary_generator.generate_summary()
        
        print(f"Engagement summary saved to {summary_file}")
        print(f"All results saved to {self.output_dir}")
        
        return summary_file

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Classroom Behavior Analysis")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", default=config.MODEL_PATH, help="Path to YOLOv8 model weights")
    parser.add_argument("--output", default=config.OUTPUT_DIR, help="Output directory")
    
    args = parser.parse_args()
    
    # Create and run behavior analysis system
    behavior_system = BehaviorAnalysisSystem(args.model, args.video, args.output)
    summary_file = behavior_system.process_video()
    
    print(f"To view results in dashboard, run: streamlit run prof_app.py")

if __name__ == "__main__":
    main() 