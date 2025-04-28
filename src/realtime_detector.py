"""
Real-time Object Detection Module
================================

This module provides functionality for real-time object detection using webcam input.
It's designed to work in different environments, including Google Colab and Kaggle.
"""

import os
import sys
import time
import threading
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML
import base64
from io import BytesIO
from PIL import Image
import queue
from typing import Union, List, Dict, Tuple, Optional, Callable, Any

# Determine if running in Google Colab
IN_COLAB = 'google.colab' in sys.modules
# Determine if running in Kaggle
IN_KAGGLE = 'kaggle_secrets' in sys.modules
# Determine if running in a Jupyter notebook
IN_NOTEBOOK = 'ipykernel' in sys.modules

class RealTimeDetector:
    """
    Class for real-time object detection using webcam input.
    
    This class provides methods for capturing webcam input and performing
    real-time object detection using YOLOv8 or other models. It's designed
    to work in different environments, including Google Colab and Kaggle.
    """
    
    def __init__(
        self, 
        detector=None, 
        input_size: Tuple[int, int] = (640, 480),
        fps_limit: int = 30,
        buffer_size: int = 5
    ):
        """
        Initialize the RealTimeDetector.
        
        Args:
            detector: Object detector, specifically YOLOv8Detector
            input_size: Tuple of (width, height) for input size
            fps_limit: Maximum frames per second to process
            buffer_size: Size of the frame buffer for smooth processing
        """
        self.detector = detector
        self.input_size = input_size
        self.fps_limit = fps_limit
        self.min_frame_time = 1.0 / fps_limit
        self.buffer_size = buffer_size
        
        # Frame processing attributes
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.latest_frame = None
        self.processed_frame = None
        self.running = False
        self.capture_thread = None
        self.process_thread = None
        
        # Performance metrics
        self.fps = 0
        self.processing_time = 0
        self.frame_count = 0
        self.start_time = 0
        
        # Detection results
        self.detection_results = []
        self.aggregated_results = {}
        
        # Environment info
        self.env_info = self._get_environment_info()
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """
        Get information about the current environment.
        
        Returns:
            Dictionary with environment information
        """
        info = {
            'in_colab': IN_COLAB,
            'in_kaggle': IN_KAGGLE,
            'in_notebook': IN_NOTEBOOK,
            'has_cv2_gui': not (IN_COLAB or IN_KAGGLE),
            'platform': sys.platform
        }
        
        # Check if we can use cv2.imshow
        if not (IN_COLAB or IN_KAGGLE):
            try:
                # Try creating a small window and destroying it
                cv2.namedWindow("test", cv2.WINDOW_NORMAL)
                cv2.destroyWindow("test")
                info['cv2_gui_works'] = True
            except Exception:
                info['cv2_gui_works'] = False
        else:
            info['cv2_gui_works'] = False
        
        return info
    
    def _capture_frames_opencv(self, camera_id: int = 0):
        """
        Capture frames using OpenCV's VideoCapture.
        
        Args:
            camera_id: Camera ID (usually 0 for the default camera)
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            self.running = False
            return
        
        # Set camera properties if possible
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.input_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.input_size[1])
        
        last_frame_time = time.time()
        
        try:
            while self.running:
                # Limit frame rate
                current_time = time.time()
                time_since_last_frame = current_time - last_frame_time
                if time_since_last_frame < self.min_frame_time:
                    time.sleep(self.min_frame_time - time_since_last_frame)
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Update last frame time
                last_frame_time = time.time()
                
                # Store the latest frame
                self.latest_frame = frame
                
                # Add to buffer if not full
                if not self.frame_buffer.full():
                    self.frame_buffer.put(frame)
                else:
                    # Skip frames if buffer is full
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.put(frame)
                    except queue.Empty:
                        pass
        
        finally:
            cap.release()
            print("Camera released")
    
    def _capture_frames_js(self):
        """
        Capture frames using JavaScript in Colab/Jupyter.
        This method doesn't actually capture frames directly,
        but sets up the infrastructure to receive frames from JavaScript.
        """
        if not IN_COLAB:
            print("JavaScript capture is only supported in Google Colab")
            return
        
        # Import Colab-specific modules
        from google.colab.output import eval_js
        from IPython.display import display, Javascript
        
        # Set up JavaScript to capture webcam frames
        js_code = """
        async function setupWebcam() {
            const video = document.createElement('video');
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            video.srcObject = stream;
            await video.play();
            
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const context = canvas.getContext('2d');
            
            // Display a preview of the webcam feed
            const previewCanvas = document.createElement('canvas');
            previewCanvas.width = canvas.width;
            previewCanvas.height = canvas.height;
            previewCanvas.style.border = '1px solid black';
            previewCanvas.style.width = '320px';  // Smaller display size
            previewCanvas.style.height = 'auto';
            document.body.appendChild(previewCanvas);
            const previewContext = previewCanvas.getContext('2d');
            
            const sendFrame = async () => {
                if (video.videoWidth > 0) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    // Update preview
                    previewContext.drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);
                    
                    if (window.captureActive) {
                        const imageData = canvas.toDataURL('image/jpeg', 0.8);
                        
                        // Send to Python
                        google.colab.kernel.invokeFunction('webcam_frame', [imageData], {});
                    }
                    
                    // Request next frame
                    setTimeout(sendFrame, 33);  // Approx 30 fps
                } else {
                    // Video not ready yet, wait and try again
                    setTimeout(sendFrame, 100);
                }
            };
            
            window.captureActive = true;
            window.stopWebcam = () => {
                window.captureActive = false;
                stream.getTracks().forEach(track => track.stop());
                if (previewCanvas.parentNode) {
                    previewCanvas.parentNode.removeChild(previewCanvas);
                }
            };
            
            sendFrame();
            return 'Webcam started';
        }
        
        setupWebcam();
        """
        
        # Display the JavaScript code
        display(Javascript(js_code))
        
        # Define a callback function to receive frames from JavaScript
        def on_webcam_frame(frame_data):
            if not self.running:
                return
                
            if not frame_data:
                return
                
            try:
                # The frame comes as a data URL (base64 encoded)
                image_data = frame_data.split(',')[1]
                binary_data = base64.b64decode(image_data)
                
                # Convert to numpy array
                image_array = np.frombuffer(binary_data, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Store the latest frame
                    self.latest_frame = frame
                    
                    # Add to buffer if not full
                    if not self.frame_buffer.full():
                        self.frame_buffer.put(frame)
                    else:
                        # Skip frames if buffer is full
                        try:
                            self.frame_buffer.get_nowait()
                            self.frame_buffer.put(frame)
                        except queue.Empty:
                            pass
            except Exception as e:
                print(f"Error processing webcam frame: {e}")
        
        # Register the callback function
        from google.colab.output import register_callback
        register_callback('webcam_frame', on_webcam_frame)
        
        # Keep the thread alive while running
        while self.running:
            time.sleep(0.1)
        
        # Stop the webcam when done
        try:
            eval_js("window.stopWebcam()")
        except Exception:
            pass
    
    def _process_frames(self):
        """
        Process frames from the buffer using the detector.
        """
        if self.detector is None:
            print("Error: No detector provided")
            self.running = False
            return
        
        self.start_time = time.time()
        self.frame_count = 0
        
        while self.running:
            try:
                # Get a frame from the buffer
                frame = self.frame_buffer.get(timeout=1.0)
                
                # Measure processing time
                start_process = time.time()
                
                # Run detection on the frame
                result = self.detector.detect(frame, show_result=False)
                
                # Calculate processing time
                self.processing_time = time.time() - start_process
                
                # Draw detection results on the frame
                processed_frame = result.plot()
                
                # Store the processed frame and result
                self.processed_frame = processed_frame
                self.detection_results.append(result)
                
                # Update metrics
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.fps = self.frame_count / elapsed
                
                # Update aggregated results
                self._update_aggregated_results(result)
                
                # Limit the number of stored results
                if len(self.detection_results) > 30:  # Keep only last 30 results
                    self.detection_results.pop(0)
                
                # Signal that we've processed the frame
                self.frame_buffer.task_done()
                
            except queue.Empty:
                # No frames in buffer, wait a bit
                time.sleep(0.01)
            except Exception as e:
                print(f"Error processing frame: {e}")
                time.sleep(0.1)
    
    def _update_aggregated_results(self, result):
        """
        Update aggregated detection results.
        
        Args:
            result: Detection result from the detector
        """
        boxes = result.boxes
        
        # Count detections by class
        classes = {}
        for box in boxes:
            class_id = int(box.cls.item())
            class_name = self.detector.class_names[class_id]
            confidence = float(box.conf.item())
            
            if class_name in classes:
                classes[class_name]['count'] += 1
                classes[class_name]['confidences'].append(confidence)
            else:
                classes[class_name] = {
                    'count': 1,
                    'confidences': [confidence]
                }
        
        # Update aggregated results
        for class_name, data in classes.items():
            if class_name in self.aggregated_results:
                self.aggregated_results[class_name]['total_count'] += data['count']
                self.aggregated_results[class_name]['confidences'].extend(data['confidences'])
                self.aggregated_results[class_name]['frames_detected'] += 1
            else:
                self.aggregated_results[class_name] = {
                    'total_count': data['count'],
                    'confidences': data['confidences'],
                    'frames_detected': 1
                }
        
        # Update frame count for all classes
        for class_name in self.aggregated_results:
            if class_name not in classes:
                # Class not detected in this frame
                self.aggregated_results[class_name]['frames_detected'] += 0
    
    def start(self, camera_id: int = 0, use_js: bool = None):
        """
        Start real-time object detection.
        
        Args:
            camera_id: Camera ID for OpenCV capture
            use_js: Whether to use JavaScript for capture in Colab
                   (if None, will use JS in Colab, OpenCV otherwise)
        """
        if self.running:
            print("Real-time detection is already running")
            return
        
        # Reset attributes
        self.frame_buffer = queue.Queue(maxsize=self.buffer_size)
        self.latest_frame = None
        self.processed_frame = None
        self.detection_results = []
        self.aggregated_results = {}
        self.fps = 0
        self.processing_time = 0
        self.frame_count = 0
        
        # Set running flag
        self.running = True
        
        # Determine whether to use JavaScript capture
        if use_js is None:
            use_js = IN_COLAB
        
        # Start capture thread
        if use_js and IN_COLAB:
            print("Starting webcam capture using JavaScript...")
            self.capture_thread = threading.Thread(target=self._capture_frames_js)
        else:
            print(f"Starting webcam capture using OpenCV (camera_id={camera_id})...")
            self.capture_thread = threading.Thread(target=self._capture_frames_opencv, args=(camera_id,))
        
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        print("Real-time object detection started")
        
        # Give time for initialization
        time.sleep(1.0)
    
    def stop(self):
        """Stop real-time object detection."""
        if not self.running:
            print("Real-time detection is not running")
            return
        
        # Set running flag to False
        self.running = False
        
        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
        
        # Reset JavaScript capture if running in Colab
        if IN_COLAB:
            try:
                from google.colab.output import eval_js
                eval_js("if (window.stopWebcam) window.stopWebcam()")
            except Exception:
                pass
        
        print("Real-time object detection stopped")
        
        # Print results summary
        self.print_results_summary()
    
    def get_latest_frame(self):
        """Get the latest captured frame."""
        return self.latest_frame
    
    def get_processed_frame(self):
        """Get the latest processed frame with detection results."""
        return self.processed_frame
    
    def print_results_summary(self):
        """Print a summary of detection results."""
        if not self.aggregated_results:
            print("No detection results available")
            return
        
        print("\nDetection Results Summary:")
        print(f"Processed {self.frame_count} frames at {self.fps:.2f} FPS")
        print(f"Average processing time: {self.processing_time*1000:.2f} ms per frame")
        print("\nObjects detected:")
        
        # Sort by frequency
        sorted_results = sorted(
            self.aggregated_results.items(),
            key=lambda x: x[1]['total_count'],
            reverse=True
        )
        
        for class_name, data in sorted_results:
            avg_confidence = sum(data['confidences']) / len(data['confidences'])
            max_confidence = max(data['confidences'])
            
            print(f"  {class_name}:")
            print(f"    Count: {data['total_count']}")
            print(f"    Detected in {data['frames_detected']} frames")
            print(f"    Average confidence: {avg_confidence:.2f}")
            print(f"    Max confidence: {max_confidence:.2f}")
    
    def display_live_feed(self, update_interval: float = 0.1, max_frames: int = None):
        """
        Display the live feed with detection results in a Jupyter notebook.
        
        Args:
            update_interval: Time between display updates (seconds)
            max_frames: Maximum number of frames to display (None for unlimited)
        """
        if not IN_NOTEBOOK:
            print("Live feed display is only supported in Jupyter notebooks")
            return
        
        frames_displayed = 0
        
        try:
            while self.running and (max_frames is None or frames_displayed < max_frames):
                if self.processed_frame is not None:
                    # Convert BGR to RGB for display
                    rgb_frame = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Clear previous output
                    clear_output(wait=True)
                    
                    # Display the frame
                    plt.figure(figsize=(12, 8))
                    plt.imshow(rgb_frame)
                    plt.title(f"FPS: {self.fps:.2f}, Processing time: {self.processing_time*1000:.2f} ms")
                    plt.axis('off')
                    plt.show()
                    
                    frames_displayed += 1
                
                # Wait before next update
                time.sleep(update_interval)
        
        except KeyboardInterrupt:
            print("Live feed display stopped by user")
        
        finally:
            # Ensure we stop detection when display ends
            if self.running:
                self.stop()
    
    def save_video(self, output_path: str, duration: float = 10.0, fps: int = 30):
        """
        Save the detection results as a video file.
        
        Args:
            output_path: Path to save the video file
            duration: Duration to record in seconds
            fps: Frames per second for the output video
        """
        if not self.running:
            print("Error: Real-time detection is not running")
            return
        
        print(f"Recording video to {output_path} for {duration} seconds...")
        
        # Get frame dimensions from the first processed frame
        while self.processed_frame is None:
            time.sleep(0.1)
            if not self.running:
                print("Error: Detection stopped before recording could start")
                return
        
        height, width = self.processed_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Record for the specified duration
        start_time = time.time()
        frames_recorded = 0
        
        try:
            while self.running and (time.time() - start_time) < duration:
                if self.processed_frame is not None:
                    # Write the frame to the video
                    out.write(self.processed_frame)
                    frames_recorded += 1
                
                # Wait for next frame
                time.sleep(1.0 / fps)
        
        finally:
            # Release the video writer
            out.release()
            print(f"Video saved to {output_path} ({frames_recorded} frames)")
    
    def run_display_loop(self, display_mode: str = 'notebook'):
        """
        Run a display loop for the live feed.
        
        Args:
            display_mode: Display mode ('notebook', 'opencv', or 'none')
        """
        if not self.running:
            print("Error: Real-time detection is not running")
            return
        
        if display_mode == 'notebook' and IN_NOTEBOOK:
            # Use Jupyter notebook display
            self.display_live_feed()
        
        elif display_mode == 'opencv' and not (IN_COLAB or IN_KAGGLE):
            # Use OpenCV window display
            cv2.namedWindow("Real-time Object Detection", cv2.WINDOW_NORMAL)
            
            try:
                while self.running:
                    if self.processed_frame is not None:
                        # Add FPS and processing time to the frame
                        frame = self.processed_frame.copy()
                        fps_text = f"FPS: {self.fps:.2f}"
                        time_text = f"Processing: {self.processing_time*1000:.2f} ms"
                        
                        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, time_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Display the frame
                        cv2.imshow("Real-time Object Detection", frame)
                        
                        # Check for key press to exit
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' or ESC
                            break
                    
                    # Wait before checking for the next frame
                    time.sleep(0.01)
            
            finally:
                cv2.destroyAllWindows()
                if self.running:
                    self.stop()
        
        else:
            # No display, just run detection
            print("Running without display. Press Ctrl+C to stop.")
            try:
                while self.running:
                    time.sleep(0.1)
                    
                    # Print status every 5 seconds
                    if self.frame_count % (5 * int(self.fps)) == 0:
                        print(f"FPS: {self.fps:.2f}, Frames: {self.frame_count}")
            
            except KeyboardInterrupt:
                print("Detection stopped by user")
            
            finally:
                if self.running:
                    self.stop()


# If the YOLOv8 detector is available, import it
try:
    from .yolo_detector import YOLOv8Detector
except ImportError:
    try:
        from yolo_detector import YOLOv8Detector
    except ImportError:
        print("Warning: YOLOv8Detector not found, you'll need to provide a detector when using RealTimeDetector")


def init_detector_for_realtime(model_size='n', conf=0.25, iou=0.45):
    """
    Initialize a YOLOv8 detector for real-time detection.
    
    Args:
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        conf: Confidence threshold
        iou: IoU threshold
        
    Returns:
        YOLOv8Detector instance
    """
    try:
        detector = YOLOv8Detector(model_size=model_size, conf=conf, iou=iou)
        return detector
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return None


def test_webcam_availability():
    """
    Test if a webcam is available and working.
    
    Returns:
        bool: True if webcam is available, False otherwise
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam detected")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print("Webcam detected but could not capture frame")
        return False
    
    print(f"Webcam detected and working (frame size: {frame.shape[1]}x{frame.shape[0]})")
    return True


def run_detection_demo(detector=None, duration=30):
    """
    Run a real-time object detection demo.
    
    Args:
        detector: Object detector (if None, will create a YOLOv8 detector)
        duration: Demo duration in seconds
    """
    # Create detector if not provided
    if detector is None:
        detector = init_detector_for_realtime()
        if detector is None:
            print("Error: Could not initialize detector")
            return
    
    # Create real-time detector
    realtime_detector = RealTimeDetector(detector)
    
    # Start detection
    realtime_detector.start()
    
    # Determine display mode based on environment
    if IN_NOTEBOOK:
        display_mode = 'notebook'
    elif not (IN_COLAB or IN_KAGGLE) and realtime_detector.env_info.get('cv2_gui_works', False):
        display_mode = 'opencv'
    else:
        display_mode = 'none'
    
    # Set up a timer to stop detection after the specified duration
    stop_timer = threading.Timer(duration, realtime_detector.stop)
    stop_timer.daemon = True
    stop_timer.start()
    
    # Run display loop
    try:
        realtime_detector.run_display_loop(display_mode)
    except KeyboardInterrupt:
        print("Demo stopped by user")
    finally:
        # Ensure detection is stopped
        if realtime_detector.running:
            realtime_detector.stop()
        
        # Cancel timer if still active
        stop_timer.cancel()


if __name__ == "__main__":
    # Test the module with a simple demo
    print("Testing real-time object detection module...")
    
    # Test webcam availability
    if not test_webcam_availability():
        print("Webcam not available, demo cannot run")
        sys.exit(1)
    
    # Initialize detector
    detector = init_detector_for_realtime()
    if detector is None:
        print("Error initializing detector, demo cannot run")
        sys.exit(1)
    
    # Run the demo
    run_detection_demo(detector, duration=10)
    print("Demo completed")