"""
COCO Dataset Utilities for Object Detection with YOLOv8
======================================================

This module provides utilities for downloading, processing, and managing the COCO dataset
for object detection tasks.

Functions:
    - download_coco_dataset: Download and extract COCO dataset
    - convert_coco_to_yolo: Convert COCO annotations to YOLO format
    - create_dataset_yaml: Create YAML configuration for YOLOv8 training
    - visualize_coco_annotations: Visualize COCO annotations on images
"""

import os
import json
import requests
import shutil
from tqdm import tqdm
import zipfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


def download_coco_dataset(root_dir="coco_dataset", download_train=False, download_val=True):
    """
    Download and extract COCO dataset
    
    Args:
        root_dir (str): Directory to save the dataset
        download_train (bool): Whether to download training data (large ~19GB)
        download_val (bool): Whether to download validation data
        
    Returns:
        None
    """
    os.makedirs(root_dir, exist_ok=True)
    
    # Download annotation files
    if not os.path.exists(os.path.join(root_dir, 'annotations')):
        print("Downloading COCO annotations...")
        url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        download_and_extract_zip(url, os.path.join(root_dir, "annotations.zip"), root_dir)
        os.remove(os.path.join(root_dir, "annotations.zip"))
    else:
        print("COCO annotations already exist.")
        
    # Download validation images
    if download_val and not os.path.exists(os.path.join(root_dir, 'val2017')):
        print("Downloading COCO validation images...")
        url = "http://images.cocodataset.org/zips/val2017.zip"
        download_and_extract_zip(url, os.path.join(root_dir, "val2017.zip"), root_dir)
        os.remove(os.path.join(root_dir, "val2017.zip"))
    elif download_val:
        print("COCO validation images already exist.")
        
    # Download training images (optional, large download)
    if download_train and not os.path.exists(os.path.join(root_dir, 'train2017')):
        print("Downloading COCO training images (large ~19GB)...")
        url = "http://images.cocodataset.org/zips/train2017.zip"
        download_and_extract_zip(url, os.path.join(root_dir, "train2017.zip"), root_dir)
        os.remove(os.path.join(root_dir, "train2017.zip"))
    elif download_train:
        print("COCO training images already exist.")


def download_and_extract_zip(url, save_path, extract_dir):
    """
    Download and extract a zip file
    
    Args:
        url (str): URL to download from
        save_path (str): Path to save the downloaded file
        extract_dir (str): Directory to extract the zip file
        
    Returns:
        None
    """
    # Download the file
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get('content-length', 0))
        with open(save_path, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(save_path)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
    
    # Extract the zip file
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc=f"Extracting {os.path.basename(save_path)}"):
            zip_ref.extract(member, extract_dir)


def convert_coco_to_yolo(coco_dir, output_dir, dataset_type='val2017'):
    """
    Convert COCO annotations to YOLO format
    
    Args:
        coco_dir (str): Directory containing COCO dataset
        output_dir (str): Directory to save YOLO format data
        dataset_type (str): Dataset type ('train2017' or 'val2017')
        
    Returns:
        None
    """
    # Create output directories
    images_dir = os.path.join(output_dir, 'images', dataset_type)
    labels_dir = os.path.join(output_dir, 'labels', dataset_type)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Initialize COCO API
    ann_file = os.path.join(coco_dir, 'annotations', f'instances_{dataset_type}.json')
    coco = COCO(ann_file)
    
    # Get all categories and create a mapping to YOLO indices (0-indexed)
    categories = coco.loadCats(coco.getCatIds())
    category_id_to_yolo_id = {cat['id']: i for i, cat in enumerate(categories)}
    
    # Save category mapping for reference
    with open(os.path.join(output_dir, 'coco_to_yolo_categories.json'), 'w') as f:
        mapping = {
            'categories': [
                {'coco_id': cat['id'], 'yolo_id': i, 'name': cat['name']}
                for i, cat in enumerate(categories)
            ]
        }
        json.dump(mapping, f, indent=2)
    
    # Process all images
    img_ids = coco.getImgIds()
    print(f"Converting {len(img_ids)} {dataset_type} images to YOLO format...")
    
    for img_id in tqdm(img_ids):
        # Get image info
        img_info = coco.loadImgs(img_id)[0]
        img_file = img_info['file_name']
        
        # Get image annotations
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Skip images without annotations
        if not anns:
            continue
        
        # Copy image file
        src_img_path = os.path.join(coco_dir, dataset_type, img_file)
        dst_img_path = os.path.join(images_dir, img_file)
        if not os.path.exists(dst_img_path):
            shutil.copy(src_img_path, dst_img_path)
        
        # Create YOLO annotation file
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        with open(label_path, 'w') as f:
            img_height, img_width = img_info['height'], img_info['width']
            
            for ann in anns:
                # Skip crowd annotations
                if ann.get('iscrowd', 0) == 1:
                    continue
                
                # Get category ID and convert to YOLO ID
                coco_cat_id = ann['category_id']
                if coco_cat_id not in category_id_to_yolo_id:
                    continue  # Skip if category is not in our mapping
                    
                yolo_cat_id = category_id_to_yolo_id[coco_cat_id]
                
                # Get bounding box in COCO format [x, y, width, height]
                x, y, w, h = ann['bbox']
                
                # Convert to YOLO format [x_center/img_width, y_center/img_height, width/img_width, height/img_height]
                # YOLO format is normalized [0-1] and centered
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                
                # Write to file: class_id x_center y_center width height
                f.write(f"{yolo_cat_id} {x_center} {y_center} {norm_width} {norm_height}\n")


def create_dataset_yaml(output_dir, num_classes):
    """
    Create YAML configuration file for YOLOv8 training
    
    Args:
        output_dir (str): Directory to save the YAML file
        num_classes (int): Number of classes in the dataset
        
    Returns:
        str: Path to the created YAML file
    """
    yaml_content = f"""
# COCO dataset for YOLOv8 training
path: {os.path.abspath(output_dir)}  # dataset root dir
train: images/train2017  # train images relative to path
val: images/val2017  # val images relative to path
test:  # test images (optional)

# Classes
names:
"""
    
    # Load class names from the mapping file
    mapping_file = os.path.join(output_dir, 'coco_to_yolo_categories.json')
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    # Add class names to YAML
    for cat in sorted(mapping['categories'], key=lambda x: x['yolo_id']):
        yaml_content += f"  {cat['yolo_id']}: {cat['name']}\n"
    
    # Write YAML file
    yaml_path = os.path.join(output_dir, 'coco.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path


def visualize_coco_annotations(coco_dir, dataset_type='val2017', num_samples=3):
    """
    Visualize COCO annotations on random images
    
    Args:
        coco_dir (str): Directory containing COCO dataset
        dataset_type (str): Dataset type ('train2017' or 'val2017')
        num_samples (int): Number of sample images to visualize
        
    Returns:
        None
    """
    # Initialize COCO API
    ann_file = os.path.join(coco_dir, 'annotations', f'instances_{dataset_type}.json')
    coco = COCO(ann_file)
    
    # Get random image IDs
    img_ids = coco.getImgIds()
    selected_img_ids = np.random.choice(img_ids, size=num_samples, replace=False)
    
    # Visualize each selected image
    for img_id in selected_img_ids:
        # Load image info and annotations
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Load the image
        img_path = os.path.join(coco_dir, dataset_type, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create a copy for drawing
        img_with_boxes = img.copy()
        
        # Draw bounding boxes and labels
        for ann in anns:
            # Skip crowd annotations
            if ann.get('iscrowd', 0) == 1:
                continue
                
            # Get bounding box coordinates
            x, y, w, h = [int(coord) for coord in ann['bbox']]
            
            # Get category name
            cat_id = ann['category_id']
            cat_name = coco.loadCats(cat_id)[0]['name']
            
            # Draw bounding box
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(img_with_boxes, cat_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display images side by side
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(img)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title('Image with Annotations')
        plt.imshow(img_with_boxes)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print annotations
        print(f"Image: {img_info['file_name']} ({img_info['width']}x{img_info['height']})")
        print(f"Number of objects: {len(anns)}")
        
        # Print details of each annotation
        for i, ann in enumerate(anns):
            if ann.get('iscrowd', 0) == 1:
                continue
                
            cat_id = ann['category_id']
            cat_name = coco.loadCats(cat_id)[0]['name']
            x, y, w, h = [int(coord) for coord in ann['bbox']]
            print(f"Object {i+1}: {cat_name}, Bbox: [x={x}, y={y}, w={w}, h={h}]")
        
        print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    # Example usage
    download_coco_dataset(download_train=False, download_val=True)
    visualize_coco_annotations('coco_dataset', num_samples=2)