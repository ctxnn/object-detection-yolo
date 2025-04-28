# YOLOv8 Object Detection Notebooks

This directory contains Jupyter notebooks for object detection using YOLOv8 with PyTorch. The notebooks are designed to work in Google Colab and Kaggle environments, as well as locally.

## Notebooks Overview

1. **01_Setup_and_COCO_Dataset_Exploration.ipynb**: Set up the environment and explore the COCO dataset structure.

2. **02_Pretrained_YOLOv8_Integration.ipynb**: Load and use pre-trained YOLOv8 models for object detection.

3. **03_Image_Upload_Object_Detection.ipynb**: Upload images and perform object detection with visualization of results.

4. **04_Real_Time_Object_Detection.ipynb**: Perform real-time object detection using a webcam.

## Running in Google Colab

To run these notebooks in Google Colab:

1. Open [Google Colab](https://colab.research.google.com/)
2. Click on `File` > `Upload notebook` and select the notebook you want to run
3. Alternatively, you can open directly from GitHub with the following links:

- [Setup and COCO Dataset Exploration](https://colab.research.google.com/github/yourusername/object-detection-yolo/blob/main/notebooks/01_Setup_and_COCO_Dataset_Exploration.ipynb)
- [Pretrained YOLOv8 Integration](https://colab.research.google.com/github/yourusername/object-detection-yolo/blob/main/notebooks/02_Pretrained_YOLOv8_Integration.ipynb)
- [Image Upload Object Detection](https://colab.research.google.com/github/yourusername/object-detection-yolo/blob/main/notebooks/03_Image_Upload_Object_Detection.ipynb)
- [Real-Time Object Detection](https://colab.research.google.com/github/yourusername/object-detection-yolo/blob/main/notebooks/04_Real_Time_Object_Detection.ipynb)

### Important Notes for Google Colab

- The first code cell in each notebook will clone the repository and install the required dependencies.
- For real-time object detection with a webcam, you'll need to grant permission for Colab to access your webcam.
- The webcam functionality works best in Chrome or Firefox browsers.

## Running in Kaggle

To run these notebooks in Kaggle:

1. Open [Kaggle](https://www.kaggle.com/)
2. Click on `Create` > `Notebook`
3. Click on `File` > `Upload Notebook` and select the notebook you want to run
4. Make sure to enable GPU acceleration by clicking on `Settings` > `Accelerator` > `GPU`

### Important Notes for Kaggle

- The first code cell in each notebook will install the required dependencies.
- For file uploads, use the Kaggle file upload interface rather than the built-in widget.
- For real-time webcam detection, you'll need to use the Kaggle webcam widget, which has some limitations compared to Colab.

## Running Locally

To run these notebooks locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/object-detection-yolo.git
   cd object-detection-yolo
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```

4. Navigate to the `notebooks` directory and open the desired notebook.

## Advanced Usage

### Using Different YOLOv8 Models

The notebooks use the YOLOv8n (nano) model by default, which is the smallest and fastest model but has lower accuracy. You can use other models by changing the `model_size` parameter when initializing the detector:

- `'n'`: YOLOv8n (nano) - Smallest and fastest
- `'s'`: YOLOv8s (small) - Good balance of speed and accuracy
- `'m'`: YOLOv8m (medium) - More accurate but slower
- `'l'`: YOLOv8l (large) - High accuracy, slow inference
- `'x'`: YOLOv8x (extra large) - Highest accuracy, slowest inference

### Adjusting Detection Parameters

You can adjust the following parameters to tune the detection:

- `conf`: Confidence threshold (0.1 to 0.9) - Higher values reduce false positives
- `iou`: IoU threshold for NMS (0.1 to 0.9) - Higher values keep more overlapping boxes

## Troubleshooting

If you encounter issues running the notebooks:

1. Make sure you have a stable internet connection for downloading models and datasets.
2. For GPU-related errors, try restarting the runtime or using CPU mode.
3. If packages are missing, manually run the setup cell to install dependencies.
4. For webcam issues in Colab, make sure you've allowed camera access in your browser.
5. If you encounter out-of-memory errors, try using a smaller batch size or model.