"""
Train YOLOv8 models on classroom behavior dataset
"""

import os
import yaml
import argparse
from ultralytics import YOLO
import config
import shutil
from datetime import datetime

def train_model(
    data_yaml_path=config.DATA_YAML_PATH,
    model_size='n',  # 'n', 's', 'm', 'l', 'x'
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    img_size=config.IMAGE_SIZE,
    device='cpu',  # Use CPU instead of GPU
    project='runs/train',
    name='yolov8_classroom'
):
    """
    Train a YOLOv8 model on custom dataset
    
    Args:
        data_yaml_path: Path to the data.yaml file
        model_size: YOLOv8 model size (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        device: Device to use (0 for first GPU, cpu for CPU)
        project: Directory to save results
        name: Experiment name
    
    Returns:
        Path to best model weights
    """
    print(f"\n{'='*50}")
    print(f"Training YOLOv8{model_size} on classroom behavior dataset")
    print(f"{'='*50}")
    print(f"Data config: {data_yaml_path}")
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Data config file not found: {data_yaml_path}")
    
    # Create a copy of data.yaml with corrected paths
    temp_yaml_path = "temp_data.yaml"
    fix_data_yaml_paths(data_yaml_path, temp_yaml_path)
    
    # Load data.yaml to check class names
    with open(temp_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Classes: {data_config.get('names', [])}")
    print(f"Number of classes: {data_config.get('nc', 0)}")
    
    # Create project directory
    os.makedirs(os.path.join(project, name), exist_ok=True)
    
    # Initialize model (optionally from pretrained checkpoint)
    if model_size in ['n', 's']:
        # For smaller models, start with pretrained weights
        model = YOLO(f'yolov8{model_size}.pt')
        print(f"Loaded pretrained YOLOv8{model_size} model")
    else:
        # For larger models, train from scratch if needed
        model = YOLO(f'yolov8{model_size}.pt')
        print(f"Loaded pretrained YOLOv8{model_size} model")
    
    # Set training arguments (optimized for accuracy)
    args = {
        'data': temp_yaml_path,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'project': project,
        'name': name,
        'patience': config.PATIENCE,      # Early stopping patience
        'optimizer': 'AdamW',             # Better optimizer
        'lr0': config.LEARNING_RATE,      # Initial learning rate
        'lrf': 0.005,                     # Final learning rate
        'momentum': 0.95,                 # Optimizer momentum
        'weight_decay': 0.0005,           # Weight decay
        'warmup_epochs': 5,               # More warmup epochs
        'warmup_momentum': 0.8,           # Warmup momentum
        'warmup_bias_lr': 0.1,            # Warmup bias lr
        'box': 7.5,                       # Box loss gain
        'cls': 0.5,                       # Cls loss gain
        'mixup': 0.2,                     # Mixup augmentation
        'mosaic': 1.0,                    # Mosaic augmentation
        'degrees': 10.0,                  # Rotation augmentation
        'translate': 0.1,                 # Translation augmentation
        'scale': 0.5,                     # Scale augmentation
        'shear': 5.0,                     # Shear augmentation
        'perspective': 0.0001,            # Perspective augmentation
        'flipud': 0.2,                    # Vertical flip augmentation
        'fliplr': 0.5,                    # Horizontal flip augmentation
        'hsv_h': 0.015,                   # HSV hue augmentation
        'hsv_s': 0.7,                     # HSV saturation augmentation
        'hsv_v': 0.4,                     # HSV value augmentation
        'save': True,                     # Save checkpoints
        'save_period': 10,                # Save checkpoint every 10 epochs
        'cache': True,                    # Cache images for faster training
        'rect': False,                    # Rectangular training with different aspect ratios
        'cos_lr': True,                   # Use cosine learning rate schedule
        'copy_paste': 0.1,                # Copy-paste augmentation
    }
    
    # Print training configuration
    print("\nTraining Configuration:")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {img_size}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Device: {device}")
    print(f"Augmentations: Enhanced (Mosaic, MixUp, Copy-Paste, etc.)")
    
    # Start training
    print("\nStarting training...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Timestamp: {timestamp}")
    
    # Train the model
    results = model.train(**args)
    
    # Clean up temporary files
    if os.path.exists(temp_yaml_path):
        os.remove(temp_yaml_path)
    
    # Print results
    print(f"\nTraining complete. Results:")
    print(f"mAP@50: {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
    print(f"Precision: {results.results_dict.get('metrics/precision(B)', 0):.3f}")
    print(f"Recall: {results.results_dict.get('metrics/recall(B)', 0):.3f}")
    
    # Get best weights path
    best_weights = os.path.join(project, name, 'weights', 'best.pt')
    print(f"Best weights saved to: {best_weights}")
    
    return best_weights

def fix_data_yaml_paths(input_yaml, output_yaml):
    """
    Fix paths in data.yaml to use absolute paths
    
    Args:
        input_yaml: Path to original data.yaml
        output_yaml: Path to save fixed data.yaml
    """
    with open(input_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    # Get parent directory of dataset
    dataset_dir = os.path.dirname(os.path.abspath(input_yaml))
    
    # Fix paths
    train_path = data.get('train', '')
    val_path = data.get('val', '')
    test_path = data.get('test', '')
    
    # Fix relative paths by converting to absolute
    if train_path and train_path.startswith('..'):
        data['train'] = os.path.abspath(os.path.join(dataset_dir, train_path))
    if val_path and val_path.startswith('..'):
        data['val'] = os.path.abspath(os.path.join(dataset_dir, val_path))
    if test_path and test_path.startswith('..'):
        data['test'] = os.path.abspath(os.path.join(dataset_dir, test_path))
    
    # Save fixed data.yaml
    with open(output_yaml, 'w') as f:
        yaml.dump(data, f)
    
    print(f"Fixed data.yaml paths and saved to {output_yaml}")

def main():
    """Main entry point for training"""
    parser = argparse.ArgumentParser(description="Train YOLOv8 on classroom behavior dataset")
    parser.add_argument("--data", default=config.DATA_YAML_PATH, help="Path to data.yaml file")
    parser.add_argument("--model", default="s", choices=["n", "s", "m", "l", "x"], help="YOLOv8 model size")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=config.IMAGE_SIZE, help="Image size")
    parser.add_argument("--device", default="cpu", help="Device to use (0 for GPU, cpu for CPU)")
    parser.add_argument("--project", default="runs/train", help="Directory to save results")
    parser.add_argument("--name", default="yolov8_classroom", help="Experiment name")
    
    args = parser.parse_args()
    
    # Print startup message
    print("\n" + "="*70)
    print("Starting classroom behavior detection model training")
    print("="*70)
    print(f"Target metrics: mAP@50: >50%, Precision: >45.4%, Recall: >50.5%")
    print("="*70 + "\n")
    
    # Train model
    best_weights = train_model(
        data_yaml_path=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    model = YOLO(best_weights)
    results = model.val(data=args.data)
    
    print(f"\nValidation Results:")
    print(f"mAP@50: {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
    print(f"Precision: {results.results_dict.get('metrics/precision(B)', 0):.3f}")
    print(f"Recall: {results.results_dict.get('metrics/recall(B)', 0):.3f}")
    
    # Check if metrics meet targets
    map50 = results.results_dict.get('metrics/mAP50(B)', 0)
    precision = results.results_dict.get('metrics/precision(B)', 0)
    recall = results.results_dict.get('metrics/recall(B)', 0)
    
    if map50 >= 0.5 and precision >= 0.45 and recall >= 0.5:
        print("\n✅ SUCCESS: Model meets or exceeds target metrics!")
    else:
        print("\n⚠️ WARNING: Model does not meet all target metrics.")
        print("Consider further training or hyperparameter tuning.")
    
    print("\nNext steps:")
    print("1. Use the trained model for inference: python main.py --video path/to/video.mp4 --model " + best_weights)
    print("2. View results in dashboard: streamlit run prof_app.py")

if __name__ == "__main__":
    main() 