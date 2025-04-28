#!/usr/bin/env python3
"""
Test Script for Image Upload and Object Detection Functionality
==============================================================

This script tests the image upload and object detection functionality
implemented in the YOLOv8 object detection project.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import urllib.request

# Add the parent directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import the YOLOv8 detector
try:
    from src.yolo_detector import YOLOv8Detector
except ImportError:
    # If import fails, print error message
    print("Error: Could not import YOLOv8Detector.")
    print("Make sure you're running this script from the project root directory.")
    print("Try: python src/test_image_detection.py")
    sys.exit(1)


def download_sample_images(output_dir="sample_images"):
    """Download sample images for testing."""
    # Create directory for sample images if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List of sample images to download
    sample_images = [
        {"url": "https://ultralytics.com/images/zidane.jpg", "name": "person.jpg"},
        {"url": "https://ultralytics.com/images/bus.jpg", "name": "bus.jpg"},
        {"url": "https://raw.githubusercontent.com/ultralytics/assets/main/im/image2.jpg", "name": "people.jpg"},
        {"url": "https://raw.githubusercontent.com/ultralytics/assets/main/im/image3.jpg", "name": "traffic.jpg"}
    ]
    
    # Download each image if it doesn't exist
    downloaded_paths = []
    for img in sample_images:
        file_path = os.path.join(output_dir, img["name"])
        if not os.path.exists(file_path):
            print(f"Downloading {img['name']}...")
            try:
                urllib.request.urlretrieve(img["url"], file_path)
                downloaded_paths.append(file_path)
            except Exception as e:
                print(f"Error downloading {img['name']}: {e}")
        else:
            print(f"{img['name']} already exists.")
            downloaded_paths.append(file_path)
    
    print(f"Downloaded {len(downloaded_paths)} sample images to {output_dir}/")
    return downloaded_paths


def detect_and_display(detector, image_path):
    """Detect objects in an image and display the results."""
    print(f"\nProcessing {image_path}...")
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Perform detection
    result = detector.detect(image_path, show_result=False)
    
    # Get the detection image
    detection_img = result.plot()
    
    # Display the original and detection images side by side
    plt.figure(figsize=(15, 7))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    # Detection image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
    plt.title("Detection Results")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Print detection results
    print(f"Detection Results for {Path(image_path).name}:")
    boxes = result.boxes
    for i, box in enumerate(boxes):
        class_id = int(box.cls.item())
        class_name = detector.class_names[class_id]
        confidence = box.conf.item()
        bbox = box.xyxy[0].tolist()  # xyxy format is [x1, y1, x2, y2]
        
        print(f"  {i+1}. {class_name} (Confidence: {confidence:.2f})")
        print(f"     Bounding box: [x1={bbox[0]:.1f}, y1={bbox[1]:.1f}, x2={bbox[2]:.1f}, y2={bbox[3]:.1f}]")
    
    return result


def save_detection_results(detector, image_path, output_dir="detection_results"):
    """Save detection results to disk."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform detection
    result = detector.detect(image_path, show_result=False)
    
    # Get the detection image
    detection_img = result.plot()
    
    # Save the detection image
    output_image_path = os.path.join(output_dir, f"detection_{Path(image_path).stem}.jpg")
    cv2.imwrite(output_image_path, detection_img)
    
    # Prepare detection data for JSON
    boxes = result.boxes
    detection_data = {
        "image_name": Path(image_path).name,
        "detections": []
    }
    
    for box in boxes:
        class_id = int(box.cls.item())
        class_name = detector.class_names[class_id]
        confidence = float(box.conf.item())
        bbox = [float(x) for x in box.xyxy[0].tolist()]  # xyxy format is [x1, y1, x2, y2]
        
        detection_data["detections"].append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": bbox
        })
    
    # Save detection data to JSON
    output_json_path = os.path.join(output_dir, f"detection_{Path(image_path).stem}.json")
    with open(output_json_path, "w") as f:
        json.dump(detection_data, f, indent=2)
    
    print(f"Saved detection results to {output_image_path} and {output_json_path}")
    
    return output_image_path, output_json_path


def analyze_detection_results(detector, image_path, result):
    """Analyze the detection results and display statistics."""
    # Get detection boxes
    boxes = result.boxes
    
    if len(boxes) == 0:
        print(f"No objects detected in {Path(image_path).name}")
        return
    
    # Get class IDs and confidences
    class_ids = [int(box.cls.item()) for box in boxes]
    confidences = [box.conf.item() for box in boxes]
    class_names = [detector.class_names[class_id] for class_id in class_ids]
    
    # Count objects by class
    from collections import Counter
    class_counts = Counter(class_names)
    
    # Print detection summary
    print(f"\nDetection Summary for {Path(image_path).name}:")
    print(f"Total objects detected: {len(boxes)}")
    print("\nObjects by class:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    print(f"\nConfidence range: {min(confidences):.2f} - {max(confidences):.2f}")
    print(f"Average confidence: {sum(confidences)/len(confidences):.2f}")
    
    # Plot analysis
    plt.figure(figsize=(15, 10))
    
    # Plot class distribution
    plt.subplot(2, 2, 1)
    classes, counts = zip(*class_counts.items()) if class_counts else ([], [])
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, counts, align="center")
    plt.yticks(y_pos, classes)
    plt.xlabel("Count")
    plt.title("Object Classes")
    
    # Plot confidence distribution
    plt.subplot(2, 2, 2)
    plt.hist(confidences, bins=10, range=(0, 1))
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence Distribution")
    
    # Plot box sizes
    plt.subplot(2, 2, 3)
    box_areas = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_areas.append((x2 - x1) * (y2 - y1))
    
    plt.scatter(range(len(box_areas)), box_areas)
    plt.xlabel("Box Index")
    plt.ylabel("Box Area (pixels²)")
    plt.title("Bounding Box Sizes")
    
    # Plot detection image
    plt.subplot(2, 2, 4)
    im_array = result.plot()
    plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
    plt.title(f"Detection Results: {Path(image_path).name}")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()


def test_batch_detection(detector, image_paths):
    """Test batch detection on multiple images."""
    print(f"\nBatch processing {len(image_paths)} images...")
    
    # Perform batch detection
    batch_results = detector.detect_batch(image_paths, show_results=True)
    
    # Print summary
    print(f"Batch Detection Results:")
    for i, (path, result) in enumerate(zip(image_paths, batch_results)):
        boxes = result.boxes
        print(f"  {i+1}. {Path(path).name}: {len(boxes)} objects detected")
    
    return batch_results


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test Image Upload and Object Detection")
    parser.add_argument("-i", "--images", nargs="+", help="Paths to input images")
    parser.add_argument("-m", "--model", default="n", choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 model size (n, s, m, l, x)")
    parser.add_argument("-c", "--conf", type=float, default=0.25,
                        help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("-u", "--iou", type=float, default=0.45,
                        help="IoU threshold (0.0 to 1.0)")
    parser.add_argument("-b", "--batch", action="store_true",
                        help="Perform batch detection")
    parser.add_argument("-s", "--save", action="store_true",
                        help="Save detection results")
    parser.add_argument("-a", "--analyze", action="store_true",
                        help="Analyze detection results")
    
    args = parser.parse_args()
    
    # Initialize the YOLOv8 detector
    print(f"Initializing YOLOv8 detector (model: {args.model}, conf: {args.conf}, iou: {args.iou})...")
    detector = YOLOv8Detector(model_size=args.model, conf=args.conf, iou=args.iou)
    
    # Print model summary
    detector.print_model_summary()
    
    # Get image paths
    if args.images:
        # Use provided image paths
        image_paths = args.images
        print(f"Using {len(image_paths)} user-provided images.")
    else:
        # Download sample images
        image_paths = download_sample_images()
    
    # Check if we have valid image paths
    if not image_paths:
        print("Error: No valid image paths provided.")
        sys.exit(1)
    
    # Perform detection based on command-line arguments
    if args.batch and len(image_paths) > 1:
        # Batch detection
        batch_results = test_batch_detection(detector, image_paths)
    else:
        # Individual detection
        for path in image_paths:
            # Detect objects
            result = detect_and_display(detector, path)
            
            # Analyze detection results if requested
            if args.analyze:
                analyze_detection_results(detector, path, result)
            
            # Save detection results if requested
            if args.save:
                save_detection_results(detector, path)
    
    print("\nTesting complete!")


if __name__ == "__main__":
    main()