# Classroom Behavior Analysis System

This project uses computer vision to analyze student behavior and engagement in classroom settings using YOLOv8 object detection and DeepSORT tracking.

## Features

- 🔍 **Real-time Behavior Detection**: Identifies student behaviors including attentive, disengaged states
- 👁️ **Student Tracking**: Tracks individual students throughout the video
- 📊 **Engagement Analytics**: Generates detailed engagement metrics and visualizations
- 📱 **Professor Dashboard**: Streamlit-based dashboard for viewing engagement data

## Project Structure

- `main.py`: Core inference script that processes videos
- `tracker.py`: DeepSORT implementation for tracking students
- `logger.py`: Logs behavior data and saves face snapshots
- `summary_generator.py`: Creates engagement summary reports
- `prof_app.py`: Streamlit dashboard application
- `train.py`: Script for training/fine-tuning YOLOv8 models
- `utils/`: Utility functions and helpers

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- YOLOv8 model trained on classroom behaviors

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

#### 1. Training the Model (if needed)

```bash
python train.py
```

This will train a YOLOv8 model using the dataset in the `dataset` folder.

#### 2. Running Inference

```bash
python main.py --video path/to/classroom_video.mp4 --model path/to/model.pt
```

#### 3. Viewing the Dashboard

```bash
streamlit run prof_app.py
```

This will start the dashboard application where you can view engagement analytics.

## Model Training

The model is trained to detect the following behaviors:
- Attentive
- Disengaged (combined tired/bored behaviors)
- Other behavior states

## Output Structure

- `output/logs/`: JSON files containing behavior data
- `output/snapshots/`: Face images captured during analysis
- `output/`: Engagement summary files and annotated video 