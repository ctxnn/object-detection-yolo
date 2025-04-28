"""
YOLOv8 Detector Module
======================

This module provides a simple interface for using YOLOv8 models for object detection.
It encapsulates the Ultralytics YOLO implementation and provides convenient methods
for detecting objects in images and videos.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import time
from typing import Union, List, Dict, Tuple, Optional, Any

class YOLOv8Detector:
    """
    A class for YOLOv8 object detection.
    
    This class provides a simple interface for using YOLOv8 models for object detection
    on images and videos. It encapsulates the Ultralytics YOLO implementation and provides
    convenient methods for detecting objects.
    """
    
    def __init__(
        self, 
        model_size: str = 'n', 
        conf: float = 0.25, 
        iou: float = 0.45, 
        device: Optional[str] = None,
        custom_model_path: Optional[str] = None
    ):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            conf: Confidence threshold for detections
            iou: IoU threshold for NMS
            device: Device to use ('cuda' or 'cpu')
            custom_model_path: Path to a custom model (if not using standard YOLOv8)
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load model
        if custom_model_path:
            self.model = YOLO(custom_model_path)
            model_info = Path(custom_model_path).stem
        else:
            model_path = f'yolov8{model_size}.pt'
            self.model = YOLO(model_path)
            model_info = f"YOLOv8{model_size}"
        
        # Set detection parameters
        self.conf = conf
        self.iou = iou
        
        # Store class names
        self.class_names = self.model.names
        
        # Initialize timing variables
        self.last_inference_time = 0.0
        
        print(f"{model_info} detector initialized on {self.device}")
        print(f"Confidence threshold: {self.conf}, IoU threshold: {self.iou}")
        print(f"Model can detect {len(self.class_names)} classes")

    def detect(
        self, 
        image: Union[str, np.ndarray], 
        show_result: bool = True,
        return_processed_image: bool = False
    ) -> Any:
        """
        Perform object detection on an image.
        
        Args:
            image: Path to the input image or image array
            show_result: Whether to display the result
            return_processed_image: Whether to return the processed image
            
        Returns:
            Results object containing detections or processed image if return_processed_image=True
        """
        # Measure inference time
        start_time = time.time()
        
        # Run inference
        result = self.model(image, conf=self.conf, iou=self.iou)[0]
        
        # Record inference time
        self.last_inference_time = time.time() - start_time
        
        # Display result if requested
        if show_result:
            im_array = result.plot()
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            if isinstance(image, str):
                plt.title(f"Detection Results: {Path(image).name}")
            else:
                plt.title(f"Detection Results (Inference time: {self.last_inference_time:.3f}s)")
            plt.show()
            
            # Print detection summary
            print(f"Found {len(result.boxes)} objects (Inference time: {self.last_inference_time:.3f}s)")
            boxes = result.boxes
            for i, box in enumerate(boxes):
                class_id = int(box.cls.item())
                class_name = self.class_names[class_id]
                confidence = box.conf.item()
                bbox = box.xyxy[0].tolist()  # xyxy format is [x1, y1, x2, y2]
                
                print(f"  {i+1}. {class_name} (Confidence: {confidence:.2f})")
        
        # Return the processed image if requested
        if return_processed_image:
            return result.plot()
        
        return result
    
    def detect_batch(
        self, 
        images: List[Union[str, np.ndarray]], 
        show_results: bool = True,
        max_display: int = 10
    ) -> List[Any]:
        """
        Perform object detection on a batch of images.
        
        Args:
            images: List of paths to input images or image arrays
            show_results: Whether to display the results
            max_display: Maximum number of results to display
            
        Returns:
            List of Results objects containing detections
        """
        # Run inference on batch
        results = self.model(images, conf=self.conf, iou=self.iou)
        
        # Optionally display results
        if show_results:
            # Limit the number of results to display
            display_count = min(len(results), max_display)
            
            # Create a grid of plots
            rows = (display_count + 2) // 3  # Ceiling division
            cols = min(display_count, 3)
            
            plt.figure(figsize=(18, 6 * rows))
            
            for i in range(display_count):
                plt.subplot(rows, cols, i + 1)
                
                # Plot result
                im_array = results[i].plot()
                plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
                
                # Set title
                if isinstance(images[i], str):
                    plt.title(f"Detection: {Path(images[i]).name}")
                else:
                    plt.title(f"Detection #{i+1}")
                
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Print summary
            print(f"Detected objects in {len(results)} images:")
            for i, result in enumerate(results[:max_display]):
                print(f"\nImage {i+1}:")
                boxes = result.boxes
                for j, box in enumerate(boxes):
                    class_id = int(box.cls.item())
                    class_name = self.class_names[class_id]
                    confidence = box.conf.item()
                    
                    print(f"  {j+1}. {class_name} (Confidence: {confidence:.2f})")
            
            if len(results) > max_display:
                print(f"\n... and {len(results) - max_display} more images")
        
        return results
    
    def detect_video(
        self, 
        video_path: Union[str, int] = 0, 
        output_path: Optional[str] = None, 
        show_preview: bool = True,
        preview_width: int = 640,
        return_summary: bool = False
    ) -> Union[str, Dict[str, int]]:
        """
        Perform object detection on a video.
        
        Args:
            video_path: Path to the input video or webcam index (0 for default camera)
            output_path: Path to save the output video
            show_preview: Whether to display a preview during processing
            preview_width: Width of the preview window
            return_summary: Whether to return a summary of detections
            
        Returns:
            Path to the output video (if saved) or detection summary
        """
        # Create flag for saving
        save_video = output_path is not None
        
        # Dictionary to store detection counts
        detection_counts = {}
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if saving
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process the video
        frame_count = 0
        processing_fps = 0
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = self.model(frame, conf=self.conf, iou=self.iou)[0]
                
                # Draw results on frame
                annotated_frame = results.plot()
                
                # Calculate processing FPS
                frame_count += 1
                if frame_count % 10 == 0:  # Update FPS every 10 frames
                    processing_fps = frame_count / (time.time() - start_time)
                
                # Add FPS counter
                cv2.putText(
                    annotated_frame, 
                    f"FPS: {processing_fps:.1f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
                
                # Count detections by class
                for box in results.boxes:
                    class_id = int(box.cls.item())
                    class_name = self.class_names[class_id]
                    if class_name in detection_counts:
                        detection_counts[class_name] += 1
                    else:
                        detection_counts[class_name] = 1
                
                # Write frame to output video
                if save_video:
                    out.write(annotated_frame)
                
                # Show preview
                if show_preview:
                    # Resize for display
                    display_height = int(preview_width * height / width)
                    display_frame = cv2.resize(annotated_frame, (preview_width, display_height))
                    
                    # Show frame
                    cv2.imshow("YOLOv8 Detection", display_frame)
                    
                    # Exit on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Print progress for video file (not webcam)
                if isinstance(video_path, str) and total_frames > 0 and frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Processing: {progress:.1f}% ({frame_count}/{total_frames})", end="\r")
        
        finally:
            # Release resources
            cap.release()
            if save_video:
                out.release()
            if show_preview:
                cv2.destroyAllWindows()
            
            # Print summary
            elapsed_time = time.time() - start_time
            print(f"\nProcessed {frame_count} frames in {elapsed_time:.2f}s ({frame_count/elapsed_time:.2f} FPS)")
            
            if detection_counts:
                print("\nDetection Summary:")
                for class_name, count in sorted(detection_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {class_name}: {count} detections")
        
        # Return path or summary
        if return_summary:
            return {
                'detection_counts': detection_counts,
                'frames_processed': frame_count,
                'processing_fps': processing_fps,
                'elapsed_time': elapsed_time
            }
        else:
            return output_path if save_video else None
    
    def export_model(self, format='onnx', imgsz=640, batch=1):
        """
        Export the model to the specified format.
        
        Args:
            format: Export format (onnx, torchscript, openvino, etc.)
            imgsz: Image size for export
            batch: Batch size for export
            
        Returns:
            Path to the exported model
        """
        return self.model.export(format=format, imgsz=imgsz, batch=batch)
    
    def get_model_info(self):
        """
        Get information about the model.
        
        Returns:
            Dict with model information
        """
        # Access the underlying PyTorch model
        torch_model = self.model.model
        
        # Count parameters
        total_params = sum(p.numel() for p in torch_model.parameters())
        trainable_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
        
        # Count layer types
        layer_counts = {}
        for name, module in torch_model.named_modules():
            class_name = module.__class__.__name__
            if class_name in layer_counts:
                layer_counts[class_name] += 1
            else:
                layer_counts[class_name] = 1
        
        # Gather model info
        info = {
            'model_type': self.model.task,
            'class_names': self.class_names,
            'num_classes': len(self.class_names),
            'input_size': self.model.model.args['imgsz'] if hasattr(self.model.model, 'args') else None,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'layer_counts': layer_counts,
            'inference_device': self.device,
            'last_inference_time': self.last_inference_time
        }
        
        return info
    
    def print_model_summary(self):
        """Print a summary of the model."""
        info = self.get_model_info()
        
        print("\n" + "=" * 50)
        print(f"YOLO Model Summary")
        print("=" * 50)
        print(f"Task: {info['model_type']}")
        print(f"Number of classes: {info['num_classes']}")
        print(f"Input size: {info['input_size']}")
        print(f"Total parameters: {info['total_params']:,}")
        print(f"Trainable parameters: {info['trainable_params']:,}")
        print(f"Inference device: {info['inference_device']}")
        
        if info['last_inference_time'] > 0:
            print(f"Last inference time: {info['last_inference_time']:.3f}s")
        
        print("\nTop layer types:")
        for layer_type, count in sorted(info['layer_counts'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {layer_type}: {count}")
        
        print("\nSample classes:")
        for i, class_name in list(info['class_names'].items())[:10]:
            print(f"  {i}: {class_name}")
        
        if len(info['class_names']) > 10:
            print(f"  ... and {len(info['class_names']) - 10} more classes")
        
        print("=" * 50)


if __name__ == "__main__":
    # Example usage
    detector = YOLOv8Detector(model_size='n')
    detector.print_model_summary()
    
    # Test on a sample image if available
    try:
        import urllib.request
        
        # Download a sample image if not exists
        sample_image = "sample_image.jpg"
        if not Path(sample_image).exists():
            print("Downloading sample image...")
            url = "https://ultralytics.com/images/zidane.jpg"
            urllib.request.urlretrieve(url, sample_image)
        
        # Detect objects
        result = detector.detect(sample_image)
        
        print(f"\nDetection complete. Try running the detector with your own images or videos!")
    except Exception as e:
        print(f"Could not run sample detection: {e}")