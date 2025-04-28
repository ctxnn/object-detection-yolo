# Installation Guide

This document provides instructions for installing and setting up the Object Detection using Pre-trained YOLO Models project.

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.7 or newer
- pip package manager
- (Optional) Virtual environment tool like venv or conda

## Installation Options

### Option 1: Using pip

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using the setup script

1. Make the setup script executable:
   ```bash
   chmod +x setup_environment.sh
   ```

2. Run the setup script:
   ```bash
   ./setup_environment.sh
   ```

### Option 3: Manual installation of key packages

If you prefer to install packages individually:

```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install matplotlib
pip install jupyter ipywidgets
```

## Google Colab / Kaggle Setup

For Google Colab or Kaggle notebooks, simply include these installation commands at the beginning of your notebook:

```python
!pip install ultralytics
!pip install opencv-python
```

The notebooks in this project already include these installation commands where needed.

## Verifying Installation

To verify that your installation is working correctly:

1. Run the environment check script:
   ```bash
   python src/check_environment.py
   ```

2. Or open and run the first notebook:
   ```bash
   jupyter notebook notebooks/01_Setup_and_COCO_Dataset_Exploration.ipynb
   ```

## Troubleshooting

### Common Issues

1. **Package conflicts**:
   - Try installing in a fresh virtual environment:
     ```bash
     python -m venv yolo-env
     source yolo-env/bin/activate  # On Windows: yolo-env\Scripts\activate
     pip install -r requirements.txt
     ```

2. **OpenCV installation issues**:
   - On Linux, you might need additional system packages:
     ```bash
     sudo apt-get update
     sudo apt-get install -y libgl1-mesa-glx
     pip install opencv-python
     ```

3. **PyTorch installation issues**:
   - Visit [PyTorch installation guide](https://pytorch.org/get-started/locally/) to get the correct command for your OS and CUDA version.

4. **Permission errors**:
   - Use `sudo` on Linux/Mac or run as administrator on Windows
   - Or install in user mode: `pip install --user -r requirements.txt`

## Next Steps

After installation:

1. Explore the notebooks in the `/notebooks` directory
2. Read the documentation in the `/docs` directory
3. Try the demo applications in the `/demos` directory

For detailed usage instructions, refer to the main [README.md](README.md) and the [Implementation Guide](docs/Implementation_Guide.md).