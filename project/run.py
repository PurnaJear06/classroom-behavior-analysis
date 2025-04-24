"""
Main entry point for the Classroom Behavior Analysis System
"""

import os
import argparse
import subprocess
import sys

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import torch
        import cv2
        import ultralytics
        import streamlit
        print("All required dependencies are installed!")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError:
        print("Failed to install dependencies")
        return False

def create_directories():
    """Create required directories"""
    for directory in ["output", "output/logs", "output/snapshots"]:
        os.makedirs(directory, exist_ok=True)

def train_model(args):
    """Train YOLOv8 model"""
    from train import train_model
    
    data_yaml_path = args.data_yaml
    model_size = args.model_size
    epochs = args.epochs
    
    print(f"Training YOLOv8{model_size} model with {epochs} epochs...")
    train_model(
        data_yaml_path=data_yaml_path,
        model_size=model_size,
        epochs=epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device
    )

def run_inference(args):
    """Run inference on video"""
    from main import BehaviorAnalysisSystem
    
    video_path = args.video
    model_path = args.model
    output_dir = args.output
    
    print(f"Running inference on {video_path}...")
    system = BehaviorAnalysisSystem(model_path, video_path, output_dir)
    system.process_video()

def run_dashboard(args):
    """Run Streamlit dashboard"""
    print("Starting Streamlit dashboard...")
    try:
        subprocess.check_call([sys.executable, "-m", "streamlit", "run", "prof_app.py"])
    except subprocess.CalledProcessError:
        print("Failed to start Streamlit dashboard")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Classroom Behavior Analysis System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train YOLOv8 model")
    train_parser.add_argument("--data-yaml", required=True, help="Path to data.yaml file")
    train_parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"], help="YOLOv8 model size")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--img-size", type=int, default=640, help="Image size")
    train_parser.add_argument("--device", default="0", help="Device to use (0 for GPU, cpu for CPU)")
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Run inference on video")
    inference_parser.add_argument("--video", required=True, help="Path to input video")
    inference_parser.add_argument("--model", required=True, help="Path to model weights")
    inference_parser.add_argument("--output", default="output", help="Output directory")
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Run Streamlit dashboard")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up dependencies and directories")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        if not check_dependencies():
            if not install_dependencies():
                return
        create_directories()
        print("Setup complete!")
    
    elif args.command == "train":
        train_model(args)
    
    elif args.command == "inference":
        run_inference(args)
    
    elif args.command == "dashboard":
        run_dashboard(args)
    
    else:
        parser.print_help()
        print("\nQuick start:")
        print("1. Setup: python run.py setup")
        print("2. Train: python run.py train --data-yaml path/to/data.yaml")
        print("3. Run inference: python run.py inference --video path/to/video.mp4 --model path/to/weights.pt")
        print("4. View dashboard: python run.py dashboard")

if __name__ == "__main__":
    main() 