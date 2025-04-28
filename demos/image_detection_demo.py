#!/usr/bin/env python3
"""
Image Detection Demo Script
===========================

This script demonstrates the image upload and object detection functionality
of the YOLOv8 object detection project.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import tempfile
from PIL import Image
from io import BytesIO
import requests

# Add the parent directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import the YOLOv8 detector
try:
    from src.yolo_detector import YOLOv8Detector
except ImportError:
    print("Error: Could not import YOLOv8Detector.")
    print("Make sure you're running this script from the project root directory.")
    print("Try: python demos/image_detection_demo.py")
    sys.exit(1)


def download_sample_images(num_samples=3):
    """Download sample images for testing."""
    # Create a temporary directory for sample images
    temp_dir = tempfile.mkdtemp()
    
    # List of sample images to download
    sample_images = [
        {"url": "https://ultralytics.com/images/zidane.jpg", "name": "person.jpg"},
        {"url": "https://ultralytics.com/images/bus.jpg", "name": "bus.jpg"},
        {"url": "https://raw.githubusercontent.com/ultralytics/assets/main/im/image2.jpg", "name": "people.jpg"},
        {"url": "https://raw.githubusercontent.com/ultralytics/assets/main/im/image3.jpg", "name": "traffic.jpg"},
        {"url": "https://raw.githubusercontent.com/ultralytics/assets/main/im/image4.jpg", "name": "city.jpg"}
    ]
    
    # Download images
    downloaded_paths = []
    for i, img in enumerate(sample_images[:num_samples]):
        file_path = os.path.join(temp_dir, img["name"])
        print(f"Downloading {img['name']}...")
        
        try:
            urllib.request.urlretrieve(img["url"], file_path)
            downloaded_paths.append(file_path)
        except Exception as e:
            print(f"Error downloading {img['name']}: {e}")
    
    print(f"Downloaded {len(downloaded_paths)} sample images to {temp_dir}/")
    return downloaded_paths, temp_dir


def detect_and_display(detector, image_path):
    """Detect objects in an image and display the results."""
    print(f"Processing {image_path}...")
    
    # Measure processing time
    start_time = time.time()
    
    # Run detection
    result = detector.detect(image_path, show_result=False)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Get the detection image
    detection_img = result.plot()
    
    # Convert the image for display
    detection_img_rgb = cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB)
    
    # Get detection info
    boxes = result.boxes
    classes = [detector.class_names[int(box.cls.item())] for box in boxes]
    confidences = [float(box.conf.item()) for box in boxes]
    
    # Print detection summary
    print(f"Found {len(boxes)} objects in {processing_time:.3f} seconds:")
    for i, (class_name, confidence) in enumerate(zip(classes, confidences)):
        print(f"  {i+1}. {class_name} (Confidence: {confidence:.2f})")
    
    # Create summary string
    summary = f"Detected {len(boxes)} objects in {processing_time:.3f}s"
    
    return detection_img_rgb, summary, result


def load_image_from_url(url):
    """Load an image from a URL."""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)


def batch_process_images(detector, image_paths):
    """Process multiple images in batch."""
    print(f"Batch processing {len(image_paths)} images...")
    
    # Measure processing time
    start_time = time.time()
    
    # Process each image
    for image_path in image_paths:
        result = detector.detect(image_path, show_result=False)
        
        # Print detection summary
        boxes = result.boxes
        print(f"Image: {Path(image_path).name}, Objects: {len(boxes)}")
    
    # Calculate total processing time
    processing_time = time.time() - start_time
    
    print(f"Batch processed {len(image_paths)} images in {processing_time:.3f} seconds")
    print(f"Average time per image: {processing_time/len(image_paths):.3f} seconds")


def create_image_grid(images, titles=None, cols=3, figsize=(15, 10)):
    """Create a grid of images for display."""
    # Calculate the number of rows needed
    rows = (len(images) + cols - 1) // cols
    
    # Create the figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # If only one row, make sure axes is 2D
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each image
    for i, img in enumerate(images):
        if i < rows * cols:
            r, c = i // cols, i % cols
            ax = axes[r, c]
            
            # Display the image
            ax.imshow(img)
            
            # Set the title if provided
            if titles and i < len(titles):
                ax.set_title(titles[i])
            
            # Turn off axis
            ax.axis("off")
    
    # Turn off any remaining empty subplots
    for i in range(len(images), rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis("off")
    
    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate image upload and detection."""
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection Demo")
    parser.add_argument("-i", "--images", nargs="+", help="Paths to input images")
    parser.add_argument("-u", "--urls", nargs="+", help="URLs of input images")
    parser.add_argument("-s", "--samples", type=int, default=3, help="Number of sample images to download")
    parser.add_argument("-m", "--model", default="n", choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 model size (n, s, m, l, x)")
    parser.add_argument("-c", "--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("-b", "--batch", action="store_true", help="Enable batch processing")
    
    args = parser.parse_args()
    
    # Initialize the YOLOv8 detector
    print(f"Initializing YOLOv8{args.model} detector (conf={args.conf})...")
    detector = YOLOv8Detector(model_size=args.model, conf=args.conf)
    
    # Collect image paths from different sources
    image_paths = []
    temp_dir = None
    
    # 1. User-provided image paths
    if args.images:
        for path in args.images:
            if os.path.exists(path):
                image_paths.append(path)
            else:
                print(f"Warning: Image file not found: {path}")
    
    # 2. User-provided image URLs
    if args.urls:
        for i, url in enumerate(args.urls):
            try:
                # Create a temporary directory if not yet created
                if temp_dir is None:
                    temp_dir = tempfile.mkdtemp()
                
                # Download and save the image
                file_path = os.path.join(temp_dir, f"url_image_{i}.jpg")
                urllib.request.urlretrieve(url, file_path)
                image_paths.append(file_path)
                print(f"Downloaded image from {url}")
            except Exception as e:
                print(f"Error downloading image from {url}: {e}")
    
    # 3. Download sample images if no user images
    if not image_paths:
        print("No user images provided. Downloading sample images...")
        downloaded_paths, temp_dir = download_sample_images(args.samples)
        image_paths.extend(downloaded_paths)
    
    # Process the images
    if not image_paths:
        print("Error: No valid images found for processing.")
        sys.exit(1)
    
    if args.batch:
        # Batch processing
        batch_process_images(detector, image_paths)
    else:
        # Individual processing with visualization
        results = []
        rgb_images = []
        summaries = []
        
        for path in image_paths:
            # Load and display original image
            original_img = cv2.imread(path)
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            rgb_images.append(original_rgb)
            
            # Detect objects
            detection_img, summary, result = detect_and_display(detector, path)
            rgb_images.append(detection_img)
            summaries.append(summary)
            results.append(result)
        
        # Create titles for the grid
        titles = []
        for i, path in enumerate(image_paths):
            filename = Path(path).name
            titles.append(f"Original: {filename}")
            titles.append(f"Detection: {summaries[i]}")
        
        # Display results in a grid
        fig = create_image_grid(rgb_images, titles, cols=2)
        plt.show()
    
    # Clean up temporary directory
    if temp_dir and os.path.exists(temp_dir):
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()