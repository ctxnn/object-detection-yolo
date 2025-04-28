"""
Utility Functions for Jupyter Notebooks
======================================

This module provides utility functions for working with Jupyter notebooks,
particularly in Google Colab and Kaggle environments.
"""

import os
import sys
import tempfile
import time
from pathlib import Path
import urllib.request
import base64
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display, Javascript, clear_output
from typing import List, Dict, Tuple, Union, Optional, Any, Callable


def is_colab() -> bool:
    """Check if the code is running in Google Colab."""
    return 'google.colab' in sys.modules


def is_kaggle() -> bool:
    """Check if the code is running in Kaggle."""
    return 'kaggle_secrets' in sys.modules


def is_notebook() -> bool:
    """Check if the code is running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return False
        if 'IPKernelApp' not in get_ipython().config:
            return False
        return True
    except ImportError:
        return False


def setup_env() -> None:
    """Set up the environment based on where we're running."""
    environment = "Local"
    
    if is_colab():
        environment = "Google Colab"
    elif is_kaggle():
        environment = "Kaggle"
    
    print(f"Detected environment: {environment}")
    
    # Install required packages if needed
    if is_colab() or is_kaggle():
        print("Installing required packages...")
        try:
            # Use subprocess to avoid blocking the notebook
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "opencv-python"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ipywidgets"])
            print("✅ Packages installed successfully")
        except Exception as e:
            print(f"❌ Error installing packages: {e}")
            print("Please run: !pip install ultralytics opencv-python ipywidgets")
    
    # Check for GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {device}")
        else:
            print("⚠️ No GPU detected. Running on CPU may be slow for inference.")
    except ImportError:
        print("⚠️ Could not check GPU availability. PyTorch not installed.")


def show_upload_widget() -> None:
    """Display a file upload widget in the notebook."""
    if is_colab():
        from google.colab import files
        print("Select an image or video file to upload:")
        return files.upload()
    elif is_kaggle() or is_notebook():
        try:
            import ipywidgets as widgets
            from IPython.display import display
            
            file_upload = widgets.FileUpload(
                accept='.jpg,.jpeg,.png,.mp4,.avi,.mov',
                multiple=False,
                description='Upload:',
                button_style='primary'
            )
            display(file_upload)
            print("Use the button above to upload an image or video file.")
            return file_upload
        except ImportError:
            print("Could not create upload widget. Please install ipywidgets.")
            print("Run: !pip install ipywidgets")
    else:
        print("File upload widget is only supported in notebook environments.")
        print("Please provide the file path directly.")


def save_uploaded_file(file_content: bytes, filename: str) -> str:
    """
    Save uploaded file content to disk.
    
    Args:
        file_content: Binary content of the uploaded file
        filename: Name to use for the saved file
        
    Returns:
        Path to the saved file
    """
    # Create uploads directory if it doesn't exist
    upload_dir = Path('uploads')
    upload_dir.mkdir(exist_ok=True)
    
    # Ensure filename is safe
    safe_filename = Path(filename).name
    file_path = upload_dir / safe_filename
    
    # Save the file
    with open(file_path, 'wb') as f:
        f.write(file_content)
    
    print(f"File saved to {file_path}")
    return str(file_path)


def process_upload_widget_kaggle(file_upload) -> List[str]:
    """
    Process files from ipywidgets upload widget.
    
    Args:
        file_upload: ipywidgets.FileUpload widget with uploaded files
        
    Returns:
        List of paths to saved files
    """
    if not file_upload.value:
        print("No files uploaded.")
        return []
    
    saved_paths = []
    for filename, file_info in file_upload.value.items():
        content = file_info['content']
        path = save_uploaded_file(content, filename)
        saved_paths.append(path)
    
    return saved_paths


def process_upload_colab(uploaded_files) -> List[str]:
    """
    Process files uploaded via google.colab.files.
    
    Args:
        uploaded_files: Dictionary of uploaded files from google.colab.files.upload()
        
    Returns:
        List of paths to saved files
    """
    if not uploaded_files:
        print("No files uploaded.")
        return []
    
    saved_paths = []
    for filename, content in uploaded_files.items():
        path = save_uploaded_file(content, filename)
        saved_paths.append(path)
    
    return saved_paths


def get_video_stats(video_path: str) -> Dict[str, Any]:
    """
    Get statistics about a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video statistics
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0
    
    # Get a thumbnail
    ret, frame = cap.read()
    thumbnail = None
    if ret:
        # Resize thumbnail to a reasonable size
        max_dim = 300
        scale = max_dim / max(frame.shape[0], frame.shape[1])
        thumbnail = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    
    # Release the video capture
    cap.release()
    
    # Create stats dictionary
    stats = {
        'width': width,
        'height': height,
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'thumbnail': thumbnail
    }
    
    return stats


def display_video_info(video_path: str) -> None:
    """
    Display information about a video file.
    
    Args:
        video_path: Path to the video file
    """
    try:
        stats = get_video_stats(video_path)
        
        print(f"Video: {Path(video_path).name}")
        print(f"Resolution: {stats['width']}x{stats['height']}")
        print(f"FPS: {stats['fps']:.2f}")
        print(f"Duration: {stats['duration']:.2f} seconds ({stats['frame_count']} frames)")
        
        # Display thumbnail
        if stats['thumbnail'] is not None:
            plt.figure(figsize=(8, 4.5))
            plt.imshow(cv2.cvtColor(stats['thumbnail'], cv2.COLOR_BGR2RGB))
            plt.title("Video Thumbnail")
            plt.axis('off')
            plt.show()
    except Exception as e:
        print(f"Error getting video info: {e}")


def load_image_from_path(image_path: str) -> np.ndarray:
    """
    Load an image from a file path.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Image as a numpy array
    """
    # Check if the path exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read the image
    img = cv2.imread(image_path)
    
    # Check if the image was read successfully
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img_rgb


def display_image_with_info(image_path: str) -> None:
    """
    Display an image with its information.
    
    Args:
        image_path: Path to the image
    """
    try:
        # Load the image
        img = load_image_from_path(image_path)
        
        # Display image info
        print(f"Image: {Path(image_path).name}")
        print(f"Resolution: {img.shape[1]}x{img.shape[0]}")
        print(f"Channels: {img.shape[2]}")
        
        # Display the image
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(Path(image_path).name)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error displaying image: {e}")


def setup_webcam_stream():
    """Set up a webcam stream in Colab."""
    if not is_colab():
        print("Webcam streaming is currently only supported in Google Colab.")
        return None
    
    # This JavaScript code accesses the webcam and sends frames to Python
    display(Javascript('''
    async function setupWebcam() {
        const video = document.createElement('video');
        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        video.srcObject = stream;
        video.play();
        
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const context = canvas.getContext('2d');
        
        const sendFrame = async () => {
            if (video.videoWidth > 0) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                
                // Send to Python
                google.colab.kernel.invokeFunction('webcam_frame', [imageData], {});
                
                // Request next frame
                setTimeout(sendFrame, 100);  // 10 fps
            } else {
                // Video not ready yet, wait and try again
                setTimeout(sendFrame, 100);
            }
        };
        
        sendFrame();
    }
    
    setupWebcam();
    '''))
    
    # Create a place to store the latest frame
    webcam_data = {'latest_frame': None}
    
    # Define the callback function to receive frames from JavaScript
    def on_webcam_frame(frame_data):
        if not frame_data:
            return
            
        # The frame comes as a base64 encoded string
        image_data = frame_data.split(',')[1]
        binary_data = base64.b64decode(image_data)
        
        # Convert to numpy array
        image_array = np.frombuffer(binary_data, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # Store the latest frame
        webcam_data['latest_frame'] = frame
        
        return frame
    
    # Register the callback
    output_callback = google.colab.output.register_callback('webcam_frame', on_webcam_frame)
    
    return webcam_data


def process_webcam_frames(
    process_func: Callable, 
    stop_func: Optional[Callable] = None,
    max_frames: int = 100,
    display_fps: bool = True
):
    """
    Process frames from the webcam in real-time.
    
    Args:
        process_func: Function to process each frame
        stop_func: Function that returns True when processing should stop
        max_frames: Maximum number of frames to process
        display_fps: Whether to display FPS information
    """
    if not is_colab():
        print("Webcam streaming is currently only supported in Google Colab.")
        return
    
    from IPython.display import display, clear_output
    from google.colab.output import eval_js
    import time
    
    # Set up the webcam stream
    webcam_data = setup_webcam_stream()
    
    # Wait for webcam to initialize
    print("Initializing webcam...")
    time.sleep(3)
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    print("Processing webcam frames. Press 'q' to stop.")
    
    try:
        while frame_count < max_frames:
            # Get the latest frame
            frame = webcam_data.get('latest_frame')
            
            if frame is not None:
                # Process the frame
                processed_frame = process_func(frame)
                
                # Update FPS calculation
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                
                # Add FPS to the frame if requested
                if display_fps and processed_frame is not None:
                    cv2.putText(
                        processed_frame,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                
                # Display the processed frame
                clear_output(wait=True)
                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
                
                # Check if processing should stop
                if stop_func and stop_func():
                    print("Processing stopped by stop function.")
                    break
                
                # Sleep a bit to reduce CPU usage
                time.sleep(0.01)
            else:
                # Wait for a frame to be available
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Processing interrupted by user.")
    finally:
        print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds ({fps:.2f} FPS)")
    
    return frame_count, fps


def download_sample_images():
    """Download sample images for testing."""
    # Create directory for sample images
    sample_dir = Path('sample_images')
    sample_dir.mkdir(exist_ok=True)
    
    # List of sample images to download
    sample_images = [
        {'url': 'https://ultralytics.com/images/zidane.jpg', 'name': 'person.jpg'},
        {'url': 'https://ultralytics.com/images/bus.jpg', 'name': 'bus.jpg'},
        {'url': 'https://raw.githubusercontent.com/ultralytics/assets/main/im/image2.jpg', 'name': 'people.jpg'},
        {'url': 'https://raw.githubusercontent.com/ultralytics/assets/main/im/image3.jpg', 'name': 'traffic.jpg'}
    ]
    
    # Download each image
    downloaded_files = []
    for img in sample_images:
        file_path = sample_dir / img['name']
        if not file_path.exists():
            print(f"Downloading {img['name']}...")
            try:
                urllib.request.urlretrieve(img['url'], file_path)
                downloaded_files.append(str(file_path))
            except Exception as e:
                print(f"Error downloading {img['name']}: {e}")
        else:
            print(f"{img['name']} already exists.")
            downloaded_files.append(str(file_path))
    
    print(f"Downloaded {len(downloaded_files)} sample images to {sample_dir}/")
    return downloaded_files


if __name__ == "__main__":
    # If this module is run directly, set up the environment
    setup_env()
    print("\nThis module provides utility functions for working with Jupyter notebooks.")
    print("Import it in your notebook to use its functionality.")