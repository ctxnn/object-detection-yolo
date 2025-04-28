#!/usr/bin/env python3
"""
Real-time Object Detection Demo
==============================

This script demonstrates real-time object detection using a webcam.
It's designed to work in various environments, including local systems.
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
import threading
from pathlib import Path

# Add the parent directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import the necessary modules
try:
    from src.yolo_detector import YOLOv8Detector
    from src.realtime_detector import RealTimeDetector, test_webcam_availability
except ImportError:
    print("Error: Could not import required modules.")
    print("Make sure you're running this script from the project root directory.")
    print("Try: python demos/realtime_detection_demo.py")
    sys.exit(1)


def initialize_detector(model_size='n', conf=0.25, iou=0.45):
    """Initialize the YOLOv8 detector."""
    print(f"Initializing YOLOv8 detector (model={model_size}, conf={conf}, iou={iou})...")
    
    try:
        detector = YOLOv8Detector(model_size=model_size, conf=conf, iou=iou)
        return detector
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return None


def create_output_window(window_name="Real-time Object Detection"):
    """Create a window for displaying detection results."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)
    return window_name


def display_frame(window_name, frame, detection_info=None):
    """Display a frame in the output window."""
    # Create a copy of the frame to avoid modifying the original
    display_frame = frame.copy()
    
    # Add detection info if provided
    if detection_info:
        # Add a semi-transparent black rectangle at the top
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 80), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
        
        # Add detection info text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_frame, detection_info["fps"], (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, detection_info["objects"], (10, 60), font, 0.8, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow(window_name, display_frame)


def run_detection_demo(model_size='n', conf=0.25, iou=0.45, camera_id=0, save_video=False):
    """Run real-time object detection demo."""
    # Check webcam availability
    if not test_webcam_availability():
        print("Error: Webcam not available")
        return False
    
    # Initialize detector
    detector = initialize_detector(model_size, conf, iou)
    if not detector:
        print("Error: Could not initialize detector")
        return False
    
    # Create output window
    window_name = create_output_window()
    
    # Create real-time detector
    realtime_detector = RealTimeDetector(detector)
    
    # Set up video writer if saving video
    video_writer = None
    if save_video:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"realtime_detection_{int(time.time())}.mp4")
        
        # We'll initialize the writer after getting the first frame
        print(f"Video will be saved to: {output_path}")
    
    # Start detection
    print("Starting real-time detection...")
    print("Press 'q' to quit, 's' to save a screenshot")
    realtime_detector.start(camera_id=camera_id)
    
    # Wait for initialization
    time.sleep(1.0)
    
    # Screenshot counter
    screenshot_count = 0
    
    # Main loop
    try:
        while True:
            # Get the latest processed frame
            frame = realtime_detector.get_processed_frame()
            
            if frame is not None:
                # Initialize video writer if needed
                if save_video and video_writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
                
                # Create detection info
                detection_info = {
                    "fps": f"FPS: {realtime_detector.fps:.2f}",
                    "objects": f"Objects: {len(realtime_detector.detection_results[-1].boxes) if realtime_detector.detection_results else 0}"
                }
                
                # Display the frame
                display_frame(window_name, frame, detection_info)
                
                # Save frame to video if recording
                if save_video and video_writer is not None:
                    video_writer.write(frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                
                # Quit on 'q' or ESC
                if key == ord('q') or key == 27:
                    break
                
                # Save screenshot on 's'
                if key == ord('s'):
                    screenshot_dir = "screenshots"
                    os.makedirs(screenshot_dir, exist_ok=True)
                    screenshot_path = os.path.join(screenshot_dir, f"detection_{int(time.time())}_{screenshot_count}.jpg")
                    cv2.imwrite(screenshot_path, frame)
                    print(f"Screenshot saved to {screenshot_path}")
                    screenshot_count += 1
            
            # Short sleep to prevent high CPU usage
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    
    finally:
        # Stop detection
        realtime_detector.stop()
        
        # Release video writer if used
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to {output_path}")
        
        # Close window
        cv2.destroyAllWindows()
        
        # Print results summary
        realtime_detector.print_results_summary()
    
    return True


def main():
    """Main function for the demo."""
    parser = argparse.ArgumentParser(description="Real-time Object Detection Demo")
    parser.add_argument("-m", "--model", default="n", choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 model size (n, s, m, l, x)")
    parser.add_argument("-c", "--conf", type=float, default=0.25,
                        help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("-i", "--iou", type=float, default=0.45,
                        help="IoU threshold (0.0 to 1.0)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera ID for webcam capture (default: 0)")
    parser.add_argument("-s", "--save", action="store_true",
                        help="Save video of the detection")
    
    args = parser.parse_args()
    
    # Run the demo
    run_detection_demo(
        model_size=args.model,
        conf=args.conf,
        iou=args.iou,
        camera_id=args.camera,
        save_video=args.save
    )


if __name__ == "__main__":
    main()