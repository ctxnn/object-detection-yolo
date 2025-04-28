"""
Visualization Utilities for Object Detection Results
===================================================

This module provides functions for visualizing the results of object detection models,
including bounding boxes, labels, and confidence scores.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Tuple, Union, Optional
import random


def plot_one_box(
    box: List[float], 
    img: np.ndarray, 
    color: Tuple[int, int, int] = None, 
    label: str = None, 
    line_thickness: int = 3
) -> None:
    """
    Plot one bounding box on the image.
    
    Args:
        box: Bounding box coordinates [x1, y1, x2, y2]
        img: Image to plot on
        color: Color of the bounding box (BGR)
        label: Label to display
        line_thickness: Thickness of the bounding box lines
    """
    # Generate random color if not provided
    if color is None:
        color = [random.randint(0, 255) for _ in range(3)]
    
    # Convert to integers
    box = [int(x) for x in box]
    c1, c2 = (box[0], box[1]), (box[2], box[3])
    
    # Draw bounding box
    t = max(line_thickness - 1, 1)  # Line thickness
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
    
    # Add label if provided
    if label:
        tf = max(line_thickness - 1, 1)  # Font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # Filled rectangle for text background
        cv2.putText(
            img, 
            label, 
            (c1[0], c1[1] - 2), 
            0, 
            line_thickness / 3, 
            [225, 255, 255], 
            thickness=tf, 
            lineType=cv2.LINE_AA
        )


def visualize_detections(
    image: np.ndarray, 
    boxes: np.ndarray, 
    scores: np.ndarray, 
    class_ids: np.ndarray, 
    class_names: Dict[int, str],
    conf_threshold: float = 0.25,
    line_thickness: int = 3,
    hide_labels: bool = False,
    hide_conf: bool = False
) -> np.ndarray:
    """
    Visualize object detections on an image.
    
    Args:
        image: Input image
        boxes: Bounding boxes [N, 4] in format [x1, y1, x2, y2]
        scores: Confidence scores [N]
        class_ids: Class IDs [N]
        class_names: Dictionary mapping class IDs to class names
        conf_threshold: Confidence threshold for visualization
        line_thickness: Thickness of bounding box lines
        hide_labels: Whether to hide class labels
        hide_conf: Whether to hide confidence scores
        
    Returns:
        Image with visualized detections
    """
    # Create a copy of the image to avoid modifying the original
    img_vis = image.copy()
    
    # Define color map for visualization
    num_classes = len(class_names)
    colors = {i: [random.randint(0, 255) for _ in range(3)] for i in range(num_classes)}
    
    # Visualize each detection
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        if score < conf_threshold:
            continue
        
        # Get class name and color
        class_name = class_names.get(int(class_id), f"Class {class_id}")
        color = colors.get(int(class_id), [random.randint(0, 255) for _ in range(3)])
        
        # Prepare label
        if not hide_labels:
            label = f"{class_name}"
            if not hide_conf:
                label += f" {score:.2f}"
        else:
            label = None if hide_labels else f"{score:.2f}" if hide_labels and not hide_conf else None
        
        # Plot the box
        plot_one_box(box, img_vis, color=color, label=label, line_thickness=line_thickness)
    
    return img_vis


def visualize_batch(
    images: torch.Tensor, 
    targets: torch.Tensor = None, 
    predictions: torch.Tensor = None,
    class_names: Dict[int, str] = None,
    max_images: int = 4,
    figsize: Tuple[int, int] = (12, 12)
) -> None:
    """
    Visualize a batch of images with targets and/or predictions.
    
    Args:
        images: Batch of images [batch_size, channels, height, width]
        targets: Ground truth targets [num_targets, 6] (image_idx, class_id, x, y, w, h)
                 where x, y, w, h are normalized coordinates
        predictions: Predictions [batch_size, num_preds, 6] (class_id, conf, x1, y1, x2, y2)
                    where x1, y1, x2, y2 are absolute coordinates
        class_names: Dictionary mapping class IDs to class names
        max_images: Maximum number of images to display
        figsize: Figure size for matplotlib
    """
    # Convert images from tensor to numpy
    if isinstance(images, torch.Tensor):
        # If images have shape [batch_size, channels, height, width]
        if images.ndim == 4:
            # Clip to valid range and convert to numpy
            images_np = images.clamp(0, 1).cpu().numpy()
            
            # Convert from [batch_size, channels, height, width] to [batch_size, height, width, channels]
            images_np = np.transpose(images_np, (0, 2, 3, 1))
            
            # Convert from RGB to BGR for OpenCV
            images_np = images_np[..., ::-1].copy()
        else:
            raise ValueError("Images tensor must have 4 dimensions [batch_size, channels, height, width]")
    else:
        images_np = images
    
    # Number of images to display
    n = min(max_images, len(images_np))
    
    # Create figure and subplots
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    # Process and visualize each image
    for i in range(n):
        img = images_np[i].copy()
        
        # Scale image to 0-255 range if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        # Draw ground truth boxes (if available)
        if targets is not None:
            # Filter targets for this image
            if isinstance(targets, torch.Tensor):
                img_targets = targets[targets[:, 0] == i]  # Filter by image index
                
                # Convert normalized xywh to absolute xyxy
                h, w = img.shape[:2]
                for t in img_targets:
                    # Extract class_id, x, y, width, height
                    class_id, x, y, width, height = t[1:6].cpu().numpy()
                    class_id = int(class_id)
                    
                    # Convert to absolute coordinates
                    x1 = int((x - width / 2) * w)
                    y1 = int((y - height / 2) * h)
                    x2 = int((x + width / 2) * w)
                    y2 = int((y + height / 2) * h)
                    
                    # Get class name
                    if class_names is not None:
                        class_name = class_names.get(class_id, f"Class {class_id}")
                    else:
                        class_name = f"Class {class_id}"
                    
                    # Draw the box
                    plot_one_box(
                        [x1, y1, x2, y2], 
                        img, 
                        color=(0, 255, 0),  # Green for ground truth
                        label=class_name
                    )
        
        # Draw prediction boxes (if available)
        if predictions is not None and i < len(predictions):
            img_preds = predictions[i]
            
            if isinstance(img_preds, torch.Tensor):
                img_preds = img_preds.cpu().numpy()
            
            # Draw each prediction
            for pred in img_preds:
                if len(pred) >= 6:  # class_id, conf, x1, y1, x2, y2
                    class_id, conf = int(pred[0]), pred[1]
                    x1, y1, x2, y2 = [int(x) for x in pred[2:6]]
                    
                    # Get class name
                    if class_names is not None:
                        class_name = class_names.get(class_id, f"Class {class_id}")
                    else:
                        class_name = f"Class {class_id}"
                    
                    # Draw the box
                    plot_one_box(
                        [x1, y1, x2, y2], 
                        img, 
                        color=(255, 0, 0),  # Red for predictions
                        label=f"{class_name} {conf:.2f}"
                    )
        
        # Display the image
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Image {i}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def draw_results_grid(results, figsize=(12, 12), rows=3, cols=3):
    """
    Draw a grid of detection results from YOLOv8.
    
    Args:
        results: List of YOLOv8 Results objects
        figsize: Figure size
        rows: Number of rows
        cols: Number of columns
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (ax, result) in enumerate(zip(axes, results)):
        if i >= len(results):
            break
            
        # Plot the result
        result_img = result.plot()
        ax.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Image {i}")
        ax.axis('off')
    
    # Turn off any remaining subplots
    for i in range(len(results), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def make_grid(images, labels=None, grid_size=(2, 2), figsize=(10, 10)):
    """
    Create a grid of images with optional labels.
    
    Args:
        images: List of images to display
        labels: List of labels for each image
        grid_size: Tuple (rows, cols)
        figsize: Figure size
    """
    rows, cols = grid_size
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single row or column case
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Get the image
            img = images[i]
            
            # Convert to RGB if needed
            if img.ndim == 3 and img.shape[2] == 3:
                # Check if BGR instead of RGB
                if isinstance(img, np.ndarray):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display the image
            ax.imshow(img)
            
            # Add label if provided
            if labels is not None and i < len(labels):
                ax.set_title(labels[i])
            
            # Turn off axes
            ax.axis('off')
        else:
            # Hide unused subplots
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    # This code will only run when the module is run directly
    
    # Create a test image with a bounding box
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[:] = (240, 240, 240)  # Light gray background
    
    # Add some shapes for testing
    cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 255), -1)  # Red rectangle
    cv2.circle(img, (350, 350), 75, (0, 255, 0), -1)  # Green circle
    cv2.line(img, (50, 450), (450, 50), (255, 0, 0), 5)  # Blue line
    
    # Sample detection boxes
    boxes = np.array([
        [40, 40, 210, 210],   # Rectangle
        [275, 275, 425, 425],  # Circle
    ])
    
    # Sample scores and class IDs
    scores = np.array([0.95, 0.85])
    class_ids = np.array([0, 1])
    
    # Sample class names
    class_names = {0: "Rectangle", 1: "Circle"}
    
    # Visualize detections
    img_with_dets = visualize_detections(
        img, boxes, scores, class_ids, class_names,
        conf_threshold=0.5,
        line_thickness=2
    )
    
    # Display the result
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img_with_dets, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Object Detection Visualization Example")
    plt.tight_layout()
    plt.show()