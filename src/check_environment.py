#!/usr/bin/env python3
"""
Environment Check Script for YOLO Object Detection Project
=========================================================

This script verifies that all required libraries are installed and properly configured.
It also checks for GPU availability and displays system information.
"""

import sys
import os
import platform
import subprocess
from pathlib import Path

def print_section(title):
    """Print a section title with decorative formatting."""
    line = "=" * 80
    print(f"\n{line}")
    print(f" {title} ".center(80, "="))
    print(f"{line}\n")

def print_package_version(package_name, module_name=None):
    """Print the version of a package."""
    if module_name is None:
        module_name = package_name
    
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✓ {package_name:<20} {version}")
        return True
    except ImportError:
        print(f"✗ {package_name:<20} Not installed")
        return False

def check_system_info():
    """Check and print system information."""
    print_section("System Information")
    
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Check if running in Colab or Kaggle
    in_colab = 'google.colab' in sys.modules
    in_kaggle = 'kaggle_secrets' in sys.modules
    
    if in_colab:
        print("Environment: Google Colab")
    elif in_kaggle:
        print("Environment: Kaggle")
    else:
        print("Environment: Local")

def check_gpu_availability():
    """Check if GPU is available for PyTorch and display info."""
    print_section("GPU Information")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
            
            # Display info for each GPU
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            
            # Test CUDA operation
            print("\nTesting CUDA tensor operation:")
            x = torch.randn(10, 10).cuda()
            y = torch.randn(10, 10).cuda()
            z = x @ y
            print(f"✓ CUDA tensor operation successful")
        else:
            print("✗ CUDA is not available. The code will run on CPU only.")
    
    except ImportError:
        print("✗ PyTorch is not installed.")

def check_required_libraries():
    """Check if all required libraries are installed."""
    print_section("Required Libraries")
    
    # Core libraries
    print_package_version("PyTorch", "torch")
    print_package_version("Torchvision")
    print_package_version("Ultralytics")
    print_package_version("NumPy", "numpy")
    print_package_version("Matplotlib", "matplotlib")
    print_package_version("tqdm")
    print_package_version("Pillow", "PIL")
    print_package_version("PyYAML", "yaml")
    
    print("\n# Data processing and visualization")
    print_package_version("pycocotools")
    print_package_version("OpenCV", "cv2")
    print_package_version("Albumentations")
    print_package_version("scikit-learn")
    print_package_version("Seaborn")
    
    print("\n# Utilities")
    print_package_version("ipywidgets")
    print_package_version("tensorboard")

def check_yolo_model():
    """Check if YOLOv8 can be properly loaded."""
    print_section("YOLOv8 Model Check")
    
    try:
        from ultralytics import YOLO
        
        # Try to load a small YOLOv8 model
        print("Loading YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        
        # Print model information
        print(f"✓ Model loaded successfully")
        print(f"✓ Model type: {model.task}")
        print(f"✓ Number of classes: {len(model.names)}")
        print(f"✓ Classes: {list(model.names.values())[:5]}... (first 5)")
        
        return True
    except Exception as e:
        print(f"✗ Error loading YOLOv8 model: {str(e)}")
        return False

def check_coco_access():
    """Check if the COCO dataset can be accessed."""
    print_section("COCO Dataset Check")
    
    # Check if COCO directory exists
    coco_dir = Path("coco_dataset")
    if not coco_dir.exists():
        print("COCO dataset directory not found. The dataset will be downloaded during training.")
        return False
    
    # Check for annotation files
    ann_dir = coco_dir / "annotations"
    if not ann_dir.exists():
        print("COCO annotations directory not found.")
        return False
    
    val_ann_file = ann_dir / "instances_val2017.json"
    if not val_ann_file.exists():
        print("COCO validation annotations file not found.")
        return False
    
    # Check for image directories
    val_img_dir = coco_dir / "val2017"
    if not val_img_dir.exists():
        print("COCO validation images directory not found.")
        return False
    
    # Count images and annotations using pycocotools
    try:
        from pycocotools.coco import COCO
        
        coco = COCO(str(val_ann_file))
        num_images = len(coco.imgs)
        num_anns = len(coco.anns)
        num_cats = len(coco.cats)
        
        print(f"✓ COCO dataset accessible")
        print(f"✓ Validation images: {num_images}")
        print(f"✓ Validation annotations: {num_anns}")
        print(f"✓ Categories: {num_cats}")
        
        # List a few categories
        categories = coco.loadCats(coco.getCatIds())
        cat_names = [cat['name'] for cat in categories[:10]]
        print(f"✓ Sample categories: {cat_names}")
        
        return True
    except Exception as e:
        print(f"✗ Error accessing COCO dataset: {str(e)}")
        return False

def print_summary(checks):
    """Print a summary of all checks."""
    print_section("Summary")
    
    total_checks = len(checks)
    passed_checks = sum(1 for check in checks.values() if check)
    
    print(f"Passed: {passed_checks}/{total_checks} checks")
    
    if passed_checks == total_checks:
        print("\n✓ All checks passed! Your environment is ready for YOLO object detection.")
    else:
        print("\n✗ Some checks failed. Please install the missing dependencies or fix the issues.")
        
        # List failed checks
        print("\nFailed checks:")
        for name, passed in checks.items():
            if not passed:
                print(f"  - {name}")
        
        # Print installation instructions
        print("\nYou can install missing packages with:")
        print("pip install -r requirements.txt")

def main():
    """Run all environment checks."""
    print("Running environment checks for YOLO Object Detection project...\n")
    
    checks = {}
    
    # Run all checks
    check_system_info()
    checks["GPU availability"] = check_gpu_availability()
    checks["Required libraries"] = check_required_libraries()
    checks["YOLOv8 model"] = check_yolo_model()
    checks["COCO dataset"] = check_coco_access()
    
    # Print summary
    print_summary(checks)

if __name__ == "__main__":
    main()