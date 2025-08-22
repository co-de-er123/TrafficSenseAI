# 🚦 TrafficSense AI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.6%2B-orange)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)

A real-time AI-powered traffic monitoring system that detects and tracks vehicles using computer vision, built with Raspberry Pi, TensorFlow Lite, and AWS IoT Core.

<p align="center">
  <img src="./docs/demo.gif" alt="TrafficSense AI Demo" width="800">
</p>

## ✨ Features

- **Real-time Vehicle Detection**: Utilizes TensorFlow Lite for efficient object detection
- **Advanced Tracking**: Implements OpenCV's tracking algorithms (CSRT, KCF, MOSSE)
- **Cloud Integration**: Secure data transmission to AWS IoT Core via MQTT
- **Optimized Performance**: Designed for edge devices like Raspberry Pi
- **Customizable**: Easy to configure detection parameters and tracking settings
- **Cross-platform**: Works on Windows, Linux, and macOS for development

## 🛠️ Hardware Requirements

- Raspberry Pi 4B+ (recommended) or x86_64 system
- Raspberry Pi Camera Module or USB webcam
- 8GB+ microSD card (for Raspberry Pi)
- Internet connection (for AWS IoT integration)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/traffic-sense-ai.git
   cd traffic-sense-ai
   ```

2. **Set up a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the model**
   ```bash
   python scripts/download_model.py
   ```

### Basic Usage

1. **Run with default webcam**
   ```bash
   python src/main.py
   ```

2. **Run with advanced tracking**
   ```bash
   python examples/advanced_detection_demo.py --tracker CSRT
   ```

3. **Run with AWS IoT integration**
   ```bash
   # First, configure AWS IoT in config/aws_iot_config.py
   python src/main_iot.py
   ```

## 📊 AWS IoT Integration

1. Set up an AWS IoT Thing
2. Download certificates and update paths in `config/aws_iot_config.py`
3. Configure the IoT Policy to allow MQTT communication
4. Run with `--no-iot` flag to disable AWS IoT integration

## 🏗️ Project Structure

```
traffic-sense-ai/
├── config/               # Configuration files
│   └── aws_iot_config.py # AWS IoT settings
├── docs/                # Documentation and assets
├── examples/            # Example scripts
├── models/              # TensorFlow Lite models
├── scripts/             # Utility scripts
└── src/                 # Source code
    ├── detection/       # Object detection and tracking
    ├── iot/             # AWS IoT integration
    ├── main.py          # Main application
    └── main_iot.py      # AWS IoT enabled application
```

## 🎯 Advanced Configuration

### Tracker Types

- **CSRT**: High accuracy but slower
- **KCF**: Good balance of speed and accuracy (default)
- **MOSSE**: Very fast but less accurate

### Command Line Arguments

```
python src/main.py \
  --model models/ssd_mobilenet_v2_coco_quant_postprocess.tflite \
  --source 0 \
  --output output.mp4 \
  --no-iot
```

## 🤝 Contributing

Contributions are welcome! Please

## 📄 License

This project is licensed under the MIT License.


---

<p align="center">
  Made with ❤️ by Amruth
</p>
