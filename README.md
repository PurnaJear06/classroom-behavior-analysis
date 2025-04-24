# Classroom Behavior Analysis

An AI-powered application for analyzing classroom behavior from video footage. The system detects and tracks students, identifies behaviors, and provides insights to help teachers improve classroom engagement.

## Features

- 📊 **Real-time behavior tracking** using YOLO object detection
- 👤 **Student identification and tracking** throughout the video
- 📈 **Visualization tools** including timelines, heatmaps, and profiles
- 🧠 **AI-powered analysis** with Claude 3.7 Sonnet via Puter.js (free, no API key required)

## Directory Structure

```
classroom-behavior-analysis/
├── data/                  # Data storage directory
├── models/                # Pre-trained models
│   ├── best.pt            # YOLO model for behavior detection
│   └── res10_300x300_ssd_iter_140000.caffemodel  # Face detection model
├── src/                   # Source code
│   ├── app.py             # Main Streamlit application
│   ├── integrations/      # External API integrations
│   │   ├── claude_integration.py  # Claude AI integration
│   │   └── puter_integration.py   # Puter.js integration
│   └── tests/             # Test scripts
├── .env.example           # Example environment variables
├── requirements.txt       # Python dependencies
└── run.py                 # Main runner script
```

## Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/yourusername/classroom-behavior-analysis.git
cd classroom-behavior-analysis
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Download models:
The YOLO model (best.pt) and face detection models should be placed in the `models/` directory.

4. Run the application:
```bash
python run.py
```
or
```bash
streamlit run src/app.py
```

## Usage

1. Launch the application using the instructions above
2. Use the sidebar to configure detection settings
3. Upload a video file using the upload button
4. Click "Analyze Video" to process the footage
5. Explore the results across different tabs:
   - **Processed Video**: View the annotated video with detections
   - **Student Profiles**: View detailed profiles for each detected student
   - **Behavior Timeline**: Analyze behavior changes over time
   - **Behavior Summary**: View overall statistics and distributions
   - **AI Analysis**: Get AI-powered insights and recommendations

## AI Analysis Feature

The application includes an AI analysis feature powered by Claude 3.7 Sonnet via Puter.js. This feature:

1. Is completely free (no API key required)
2. Analyzes classroom behavior patterns
3. Identifies engagement issues
4. Provides teaching recommendations

To enable AI analysis:
1. Check "Enable AI-Powered Analysis" in the sidebar
2. Select an analysis depth (Basic, Standard, or Comprehensive)
3. Process a video to see the AI analysis results in the "AI Analysis" tab

## Technical Details

- The behavior detection uses a custom-trained YOLOv8 model
- Face detection uses OpenCV's DNN module with a pre-trained SSD model
- Student tracking uses a combination of face recognition and position-based heuristics
- Visualization is powered by Plotly and Streamlit
- AI integration uses Puter.js to access Claude 3.7 Sonnet directly in the browser

## License

[Your License Information]

## Acknowledgements

- YOLOv8 by Ultralytics
- Streamlit for the web framework
- Puter.js for AI integration
- Claude by Anthropic for AI analysis 