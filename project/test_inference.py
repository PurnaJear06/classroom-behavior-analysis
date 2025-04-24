#!/usr/bin/env python3
"""
Quick test script to verify model inference on a single image or video
"""

import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import config

def test_image(model_path, image_path, save_dir="output/test"):
    """
    Test model inference on a single image
    
    Args:
        model_path: Path to YOLOv8 model weights
        image_path: Path to test image
        save_dir: Directory to save results
    """
    print(f"Testing model on image: {image_path}")
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path)
    
    # Process results
    for i, result in enumerate(results):
        # Get original image
        img = cv2.imread(image_path)
        
        # Create annotated image
        boxes = result.boxes
        for box in boxes:
            # Get box data
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            
            # Skip low confidence detections
            if conf < config.CONFIDENCE_THRESHOLD:
                continue
            
            # Map class id to behavior
            behavior = config.CLASS_MAPPING.get(cls_id, "unknown")
            
            # Determine color based on behavior
            if behavior == "attentive":
                color = (0, 255, 0)  # Green
            elif behavior == "disengaged":
                color = (0, 0, 255)  # Red
            elif behavior == "distracted":
                color = (255, 0, 0)  # Blue
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label
            label = f"{behavior} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            
            # Draw filled rectangle for text background
            cv2.rectangle(img, (int(x1), int(y1) - text_size[1] - 5), 
                         (int(x1) + text_size[0], int(y1)), color, -1)
            
            # Draw text
            cv2.putText(img, label, (int(x1), int(y1) - 5), font, font_scale, (255, 255, 255), font_thickness)
        
        # Save annotated image
        output_path = os.path.join(save_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, img)
        print(f"Saved annotated image to: {output_path}")
        
        # Print detection statistics
        print(f"Detected {len(boxes)} objects:")
        behaviors = {}
        for box in boxes:
            cls_id = int(box.cls[0].item())
            behavior = config.CLASS_MAPPING.get(cls_id, "unknown")
            if behavior in behaviors:
                behaviors[behavior] += 1
            else:
                behaviors[behavior] = 1
                
        for behavior, count in behaviors.items():
            print(f"  - {behavior}: {count}")

def test_video(model_path, video_path, save_dir="output/test", frame_skip=5):
    """
    Test model inference on a video (processing every Nth frame)
    
    Args:
        model_path: Path to YOLOv8 model weights
        video_path: Path to test video
        save_dir: Directory to save results
        frame_skip: Process every Nth frame
    """
    print(f"Testing model on video: {video_path}")
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    output_path = os.path.join(save_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video
    frame_idx = 0
    processed_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every Nth frame
        if frame_idx % frame_skip == 0:
            # Run inference
            results = model(frame)
            
            # Process results
            result = results[0]
            boxes = result.boxes
            
            # Create annotated frame
            for box in boxes:
                # Get box data
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                
                # Skip low confidence detections
                if conf < config.CONFIDENCE_THRESHOLD:
                    continue
                
                # Map class id to behavior
                behavior = config.CLASS_MAPPING.get(cls_id, "unknown")
                
                # Determine color based on behavior
                if behavior == "attentive":
                    color = (0, 255, 0)  # Green
                elif behavior == "disengaged":
                    color = (0, 0, 255)  # Red
                elif behavior == "distracted":
                    color = (255, 0, 0)  # Blue
                else:
                    color = (128, 128, 128)  # Gray
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw label
                label = f"{behavior} {conf:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                
                # Draw filled rectangle for text background
                cv2.rectangle(frame, (int(x1), int(y1) - text_size[1] - 5), 
                             (int(x1) + text_size[0], int(y1)), color, -1)
                
                # Draw text
                cv2.putText(frame, label, (int(x1), int(y1) - 5), font, font_scale, (255, 255, 255), font_thickness)
            
            processed_frames += 1
            
            # Print progress
            if processed_frames % 10 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Processed {processed_frames} frames ({progress:.1f}%)")
        
        # Write frame to output video
        out.write(frame)
        
        # Increment frame index
        frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Processed {processed_frames} frames from video")
    print(f"Saved annotated video to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8 model inference")
    parser.add_argument("--model", default="../runs/train/yolov8_classroom/weights/best.pt", 
                        help="Path to model weights")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--video", help="Path to test video")
    parser.add_argument("--output", default="output/test", help="Output directory")
    parser.add_argument("--frame-skip", type=int, default=5, help="Process every Nth frame of video")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            return
        test_image(args.model, args.image, args.output)
    
    elif args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video not found at {args.video}")
            return
        test_video(args.model, args.video, args.output, args.frame_skip)
    
    else:
        print("Error: Please provide either --image or --video")

if __name__ == "__main__":
    main() 