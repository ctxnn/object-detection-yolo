# Object Detection using Pre-trained YOLO Models with PyTorch

![YOLO Object Detection](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png)

## Overview

This project implements real-time object detection using the YOLO (You Only Look Once) family of models, specifically YOLOv8. The system is capable of detecting and classifying multiple objects within images or video streams with high speed and accuracy.

The implementation is based on PyTorch and the Ultralytics YOLOv8 framework, providing a complete pipeline for using pre-trained models for object detection on images and real-time webcam feeds. It's designed to be easy to use, well-documented, and compatible with both Google Colab and Kaggle environments, making it accessible even without a local GPU.

## Key Features

- **Pre-trained YOLOv8 Models**: Multiple model sizes (nano to extra-large) available for different performance needs
- **Image Upload and Detection**: Upload and process images with detailed visualization of results
- **Real-time Webcam Detection**: Live object detection using webcam input, compatible with Colab and Kaggle
- **Cross-platform Compatibility**: Works seamlessly in Google Colab, Kaggle, and local environments
- **Detailed Documentation**: Comprehensive guides explaining YOLO algorithms from first principles
- **Interactive Notebooks**: Jupyter notebooks for each feature with step-by-step instructions
- **Modular Design**: Clean architecture that's easy to understand and extend

## Project Structure

```
.
├── config/                 # Configuration files
├── data/                   # Dataset directory
├── demos/                  # Demo applications
├── docs/                   # Documentation
├── models/                 # Saved model weights
├── notebooks/              # Jupyter notebooks for Colab/Kaggle
├── src/                    # Source code
├── utils/                  # Utility functions
├── README.md               # Project overview
├── requirements.txt        # Python dependencies
└── setup_environment.sh    # Environment setup script
```

## Notebooks

The project includes several Jupyter notebooks that can be run on Google Colab or Kaggle:

1. `01_Setup_and_COCO_Dataset_Exploration.ipynb`: Environment setup and COCO dataset exploration
2. `02_Pretrained_YOLOv8_Integration.ipynb`: Loading and using pre-trained YOLOv8 models
3. `03_Image_Upload_Object_Detection.ipynb`: Uploading images and performing object detection
4. `03_Image_Upload_Integration_Examples.ipynb`: Integration with other frameworks like Hugging Face and PyTorch Lightning
5. `04_Real_Time_Object_Detection.ipynb`: Real-time object detection with webcam input

## Documentation

Comprehensive documentation is provided in the `docs/` directory:

- `YOLO_Comprehensive_Guide.md`: Detailed explanation of YOLO algorithms from first principles
- `Implementation_Guide.md`: Guide to using and extending the implementation
- `Project_Structure.md`: Overview of the project organization
- `COCO_Dataset_Guide.md`: Explanation of the COCO dataset structure
- `Webcam_Guide_for_Colab_and_Kaggle.md`: Guide to using webcams in different environments

## Requirements

- Python 3.7+
- PyTorch 1.7+
- OpenCV
- Ultralytics YOLOv8
- Matplotlib
- Jupyter Notebook (for running the notebooks)

See `requirements.txt` for a complete list of dependencies.

## Quick Start

### Running on Google Colab

1. Open one of the notebooks directly in Colab:
   - [Setup and COCO Dataset Exploration](https://colab.research.google.com/github/yourusername/object-detection-yolo/blob/main/notebooks/01_Setup_and_COCO_Dataset_Exploration.ipynb)
   - [Pretrained YOLOv8 Integration](https://colab.research.google.com/github/yourusername/object-detection-yolo/blob/main/notebooks/02_Pretrained_YOLOv8_Integration.ipynb)
   - [Image Upload Object Detection](https://colab.research.google.com/github/yourusername/object-detection-yolo/blob/main/notebooks/03_Image_Upload_Object_Detection.ipynb)
   - [Real-Time Object Detection](https://colab.research.google.com/github/yourusername/object-detection-yolo/blob/main/notebooks/04_Real_Time_Object_Detection.ipynb)

2. The notebooks include all necessary setup code and will automatically download required dependencies.

### Local Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/object-detection-yolo.git
   cd object-detection-yolo
   ```

2. Run the setup script or install dependencies manually:
   ```bash
   # Option 1: Using the setup script
   ./setup_environment.sh
   
   # Option 2: Manual installation
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebooks:
   ```bash
   jupyter notebook notebooks/
   ```

4. Or use the Python scripts directly:
   ```bash
   # Image detection
   python src/test_image_detection.py --images path/to/image.jpg
   
   # Real-time detection
   python src/test_realtime_detection.py
   
   # Demo applications
   python demos/realtime_detection_demo.py
   ```

## Using the Components

### 1. YOLOv8 Detector

```python
from src.yolo_detector import YOLOv8Detector

# Initialize the detector
detector = YOLOv8Detector(
    model_size='n',  # 'n', 's', 'm', 'l', or 'x'
    conf=0.25,       # Confidence threshold
    iou=0.45         # IoU threshold
)

# Detect objects in an image
result = detector.detect('path/to/image.jpg')
```

### 2. Real-time Detection

```python
from src.yolo_detector import YOLOv8Detector
from src.realtime_detector import RealTimeDetector

# Initialize the detector
detector = YOLOv8Detector(model_size='n')

# Create a real-time detector
realtime_detector = RealTimeDetector(detector)

# Start detection
realtime_detector.start()

# Display live feed (in Jupyter notebook)
realtime_detector.display_live_feed()

# Or run with display (local environment)
realtime_detector.run_display_loop()

# Stop detection
realtime_detector.stop()
```

## Demos

The project includes several demo applications:

- `demos/image_detection_demo.py`: Command-line tool for detecting objects in images
- `demos/realtime_detection_demo.py`: GUI application for real-time detection with webcam
- `demos/image_detection_demo.html`: Web-based demo for object detection

Run a demo with:
```bash
python demos/realtime_detection_demo.py
```

## Customization

The project is designed to be easily customizable:

- Change model size for different speed/accuracy tradeoffs
- Adjust confidence thresholds for detection sensitivity
- Extend with custom post-processing
- Integrate with other frameworks and applications

See `docs/Implementation_Guide.md` for detailed customization instructions.

