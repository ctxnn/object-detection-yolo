#!/usr/bin/env python3
"""
Test Script for Real-time Object Detection
=========================================

This script tests the real-time object detection functionality
of the YOLOv8 object detection project.
"""

import os
import sys
import argparse
import time
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
    try:
        from yolo_detector import YOLOv8Detector
        from realtime_detector import RealTimeDetector, test_webcam_availability
    except ImportError:
        print("Error: Could not import YOLOv8Detector or RealTimeDetector.")
        print("Make sure you're running this script from the project root directory.")
        print("Try: python src/test_realtime_detection.py")
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


def run_realtime_detection(detector, duration=30, use_gui=False, camera_id=0):
    """Run real-time object detection."""
    # Create real-time detector
    realtime_detector = RealTimeDetector(detector)
    
    # Start detection
    print(f"Starting real-time detection for {duration} seconds...")
    realtime_detector.start(camera_id=camera_id)
    
    # Initialize display mode
    if use_gui:
        display_mode = 'opencv'
    else:
        display_mode = 'none'
    
    # Set up a timer to stop detection after the specified duration
    stop_time = time.time() + duration
    
    # Run display loop
    try:
        realtime_detector.run_display_loop(display_mode)
        
        # If not using GUI, we need to manually check for the time
        if display_mode == 'none':
            while realtime_detector.running and time.time() < stop_time:
                time.sleep(0.1)
                
                # Print status every second
                if int(time.time()) % 1 == 0:
                    fps = realtime_detector.fps
                    frame_count = realtime_detector.frame_count
                    proc_time = realtime_detector.processing_time
                    
                    print(f"FPS: {fps:.2f}, Frames: {frame_count}, Processing time: {proc_time*1000:.2f} ms", end="\r")
                
            # Stop detection if still running
            if realtime_detector.running:
                realtime_detector.stop()
    
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    finally:
        # Ensure detection is stopped
        if realtime_detector.running:
            realtime_detector.stop()
    
    return realtime_detector


def process_video_file(detector, video_path, output_path=None):
    """Process a video file with object detection."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
    
    if output_path is None:
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"processed_{Path(video_path).name}")
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Open the video
    import cv2
    import numpy as np
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process the video
    frame_idx = 0
    processing_times = []
    detection_counts = []
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            start_time = time.time()
            result = detector.detect(frame, show_result=False)
            proc_time = time.time() - start_time
            
            # Draw results on frame
            processed_frame = result.plot()
            
            # Add processing info
            avg_time = np.mean(processing_times) if processing_times else proc_time
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            
            cv2.putText(
                processed_frame,
                f"FPS: {avg_fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Write frame to output video
            out.write(processed_frame)
            
            # Update stats
            processing_times.append(proc_time)
            detection_counts.append(len(result.boxes))
            
            # Print progress
            frame_idx += 1
            if frame_idx % 10 == 0:
                progress = frame_idx / frame_count * 100
                print(f"Progress: {progress:.1f}% ({frame_idx}/{frame_count})", end="\r")
    
    finally:
        # Release resources
        cap.release()
        out.release()
    
    # Print results
    print(f"\nProcessed {frame_idx} frames")
    print(f"Average processing time: {np.mean(processing_times)*1000:.2f} ms")
    print(f"Average FPS: {1.0/np.mean(processing_times):.2f}")
    print(f"Average detections per frame: {np.mean(detection_counts):.2f}")
    print(f"Output saved to: {output_path}")
    
    return output_path


def download_sample_video():
    """Download a sample video for testing."""
    import urllib.request
    
    # Create output directory
    output_dir = 'sample_videos'
    os.makedirs(output_dir, exist_ok=True)
    
    # Download video
    url = "https://raw.githubusercontent.com/ultralytics/assets/main/DemoVideo.mp4"
    output_path = os.path.join(output_dir, "sample_video.mp4")
    
    if not os.path.exists(output_path):
        print(f"Downloading sample video from {url}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"Downloaded to {output_path}")
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None
    else:
        print(f"Using existing sample video: {output_path}")
    
    return output_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Real-time Object Detection")
    parser.add_argument("-d", "--duration", type=int, default=30,
                        help="Duration in seconds for real-time detection")
    parser.add_argument("-m", "--model", default="n", choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 model size (n, s, m, l, x)")
    parser.add_argument("-c", "--conf", type=float, default=0.25,
                        help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("-i", "--iou", type=float, default=0.45,
                        help="IoU threshold (0.0 to 1.0)")
    parser.add_argument("-g", "--gui", action="store_true",
                        help="Use OpenCV GUI for display")
    parser.add_argument("-v", "--video", type=str,
                        help="Process a video file instead of using webcam")
    parser.add_argument("-o", "--output", type=str,
                        help="Output path for processed video")
    parser.add_argument("-s", "--sample", action="store_true",
                        help="Download and process a sample video")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera ID for webcam capture (default: 0)")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = initialize_detector(model_size=args.model, conf=args.conf, iou=args.iou)
    if detector is None:
        print("Error: Could not initialize detector")
        sys.exit(1)
    
    # Process video file if specified
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        
        process_video_file(detector, args.video, args.output)
        sys.exit(0)
    
    # Download and process sample video if requested
    if args.sample:
        video_path = download_sample_video()
        if video_path:
            process_video_file(detector, video_path, args.output)
        sys.exit(0)
    
    # Check webcam availability if using real-time detection
    if not test_webcam_availability():
        print("Error: Webcam not available")
        print("You can try:")
        print("  - Checking if another application is using the webcam")
        print("  - Using a different camera ID with --camera")
        print("  - Processing a video file with --video")
        print("  - Downloading a sample video with --sample")
        sys.exit(1)
    
    # Run real-time detection
    run_realtime_detection(detector, args.duration, args.gui, args.camera)


if __name__ == "__main__":
    main()