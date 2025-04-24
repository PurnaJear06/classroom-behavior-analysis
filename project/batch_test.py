#!/usr/bin/env python3
"""
Script to batch process all test images and evaluate model performance
"""

import os
import glob
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import config
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def batch_process_images(model_path, test_dir, save_dir="output/batch_test"):
    """
    Process all images in test directory
    
    Args:
        model_path: Path to YOLOv8 model weights
        test_dir: Directory containing test images
        save_dir: Directory to save results
    """
    print(f"Batch processing images from: {test_dir}")
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "annotated"), exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Get all test images
    image_paths = glob.glob(os.path.join(test_dir, "*.jpg")) + \
                 glob.glob(os.path.join(test_dir, "*.jpeg")) + \
                 glob.glob(os.path.join(test_dir, "*.png"))
    
    print(f"Found {len(image_paths)} images")
    
    # Stats to track
    stats = {
        "total_images": len(image_paths),
        "total_detections": 0,
        "behavior_counts": defaultdict(int),
        "confidence_avg": defaultdict(list),
        "per_image_results": []
    }
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Run inference
        results = model(image_path)
        result = results[0]
        
        # Get original image
        img = cv2.imread(image_path)
        
        # Process detections
        image_stats = {
            "filename": os.path.basename(image_path),
            "detections": [],
            "behavior_counts": defaultdict(int)
        }
        
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
            
            # Update stats
            stats["total_detections"] += 1
            stats["behavior_counts"][behavior] += 1
            stats["confidence_avg"][behavior].append(conf)
            
            # Update image stats
            image_stats["behavior_counts"][behavior] += 1
            image_stats["detections"].append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(conf),
                "class_id": int(cls_id),
                "behavior": behavior
            })
            
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
        output_path = os.path.join(save_dir, "annotated", os.path.basename(image_path))
        cv2.imwrite(output_path, img)
        
        # Add image stats to overall stats
        image_stats["behavior_counts"] = dict(image_stats["behavior_counts"])
        stats["per_image_results"].append(image_stats)
    
    # Calculate average confidence per behavior
    for behavior, confidences in stats["confidence_avg"].items():
        stats["confidence_avg"][behavior] = sum(confidences) / len(confidences) if confidences else 0
    
    # Convert defaultdicts to regular dicts for JSON serialization
    stats["behavior_counts"] = dict(stats["behavior_counts"])
    stats["confidence_avg"] = dict(stats["confidence_avg"])
    
    # Save stats
    stats_path = os.path.join(save_dir, "batch_results.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Processed {len(image_paths)} images")
    print(f"Total detections: {stats['total_detections']}")
    print("Behavior counts:")
    for behavior, count in stats["behavior_counts"].items():
        print(f"  - {behavior}: {count} ({count/stats['total_detections']*100:.1f}%)")
    
    print(f"Results saved to: {stats_path}")
    
    # Generate visualizations
    generate_visualizations(stats, save_dir)
    
    return stats_path

def generate_visualizations(stats, save_dir):
    """
    Generate visualizations of batch processing results
    
    Args:
        stats: Statistics from batch processing
        save_dir: Directory to save visualizations
    """
    # Create behavior distribution pie chart
    plt.figure(figsize=(10, 6))
    behaviors = list(stats["behavior_counts"].keys())
    counts = list(stats["behavior_counts"].values())
    colors = []
    for behavior in behaviors:
        if behavior == "attentive":
            colors.append("green")
        elif behavior == "disengaged":
            colors.append("red")
        elif behavior == "distracted":
            colors.append("blue")
        else:
            colors.append("gray")
    
    plt.pie(counts, labels=behaviors, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title("Behavior Distribution Across All Test Images")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "behavior_distribution.png"))
    
    # Create confidence distribution bar chart
    plt.figure(figsize=(10, 6))
    behaviors = list(stats["confidence_avg"].keys())
    confidences = [stats["confidence_avg"][b] for b in behaviors]
    colors = []
    for behavior in behaviors:
        if behavior == "attentive":
            colors.append("green")
        elif behavior == "disengaged":
            colors.append("red")
        elif behavior == "distracted":
            colors.append("blue")
        else:
            colors.append("gray")
    
    plt.bar(behaviors, confidences, color=colors)
    plt.title("Average Confidence by Behavior")
    plt.ylabel("Confidence")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confidence_distribution.png"))
    
    # Create per-image detection count bar chart
    plt.figure(figsize=(12, 6))
    image_filenames = [result["filename"] for result in stats["per_image_results"]]
    detection_counts = [len(result["detections"]) for result in stats["per_image_results"]]
    
    plt.bar(range(len(image_filenames)), detection_counts)
    plt.title("Number of Detections per Image")
    plt.xlabel("Image")
    plt.ylabel("Number of Detections")
    plt.xticks(range(len(image_filenames)), [f"Image {i+1}" for i in range(len(image_filenames))], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "detections_per_image.png"))

def main():
    parser = argparse.ArgumentParser(description="Batch process test images and evaluate model performance")
    parser.add_argument("--model", default="../runs/train/yolov8_classroom/weights/best.pt", 
                        help="Path to model weights")
    parser.add_argument("--test-dir", default="../dataset/test/images", 
                        help="Directory containing test images")
    parser.add_argument("--output", default="output/batch_test", 
                        help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory not found at {args.test_dir}")
        return
    
    batch_process_images(args.model, args.test_dir, args.output)

if __name__ == "__main__":
    main() 