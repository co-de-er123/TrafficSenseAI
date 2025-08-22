""
Script to download a pre-trained TensorFlow Lite model for object detection.
"""

import os
import urllib.request
from pathlib import Path

def download_file(url: str, destination: str) -> None:
    """Download a file from a URL to the specified destination."""
    print(f"Downloading {url} to {destination}...")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(int(downloaded * 100 / total_size), 100)
        print(f"\rProgress: {percent}%", end="")
    
    urllib.request.urlretrieve(url, destination, show_progress)
    print("\nDownload complete!")

def main():
    """Download the SSD MobileNet V2 model."""
    # Model URL (COCO SSD MobileNet V2 Quantized)
    model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v2_1.0_quant_2018_06_29.zip"
    
    # Create models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Download the model
    zip_path = models_dir / "coco_ssd_mobilenet_v2_quant.zip"
    download_file(model_url, str(zip_path))
    
    # Extract the model (requires unzip utility)
    print("Extracting model files...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(models_dir)
    
    # Clean up the zip file
    os.remove(zip_path)
    
    print(f"Model files extracted to {models_dir}")
    print("\nNote: The model is now ready to use with TrafficSense AI!")

if __name__ == "__main__":
    main()
