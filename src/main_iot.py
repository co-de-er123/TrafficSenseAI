"""
TrafficSense AI - Main application with AWS IoT integration.
"""

import os
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Import our modules
from detection import TrafficDetector
from iot import MQTTClient, TrafficDataPublisher, AWSIoTConfig
from config.aws_iot_config import (
    AWS_IOT_ENDPOINT,
    CERTIFICATE_PATHS,
    MQTT_TOPICS,
    CLIENT_ID,
    QOS_LEVEL
)

class TrafficMonitor:
    """Main application class for traffic monitoring with IoT integration."""
    
    def __init__(self, model_path: str, iot_enabled: bool = True):
        """Initialize the traffic monitor."""
        self.detector = TrafficDetector(model_path, threshold=0.5)
        self.iot_enabled = iot_enabled
        self.mqtt_client = None
        self.publisher = None
        
        if iot_enabled:
            self._setup_iot()
    
    def _setup_iot(self) -> None:
        """Set up AWS IoT connection."""
        # Convert relative paths to absolute
        cert_dir = Path(__file__).parent.parent / "certs"
        cert_paths = {
            'root_ca': str(cert_dir / "AmazonRootCA1.pem"),
            'certificate': str(cert_dir / "device.pem.crt"),
            'private_key': str(cert_dir / "private.pem.key")
        }
        
        # Create AWS IoT config
        config = AWSIoTConfig(
            endpoint=AWS_IOT_ENDPOINT,
            client_id=CLIENT_ID,
            root_ca_path=cert_paths['root_ca'],
            cert_path=cert_paths['certificate'],
            private_key_path=cert_paths['private_key'],
            topic=MQTT_TOPICS['detections'],
            qos=QOS_LEVEL
        )
        
        # Initialize MQTT client and publisher
        self.mqtt_client = MQTTClient(config)
        if not self.mqtt_client.connect():
            print("Warning: Could not connect to AWS IoT. Running in local mode.")
            self.iot_enabled = False
        else:
            self.publisher = TrafficDataPublisher(self.mqtt_client)
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """Process a single frame and return detections."""
        # Get current timestamp
        frame_timestamp = time.time()
        
        # Run detection
        detections = self.detector.detect(frame)
        
        # Publish detections to AWS IoT
        if self.iot_enabled and self.publisher:
            detection_data = [
                {
                    'class_id': int(det['class_id']),
                    'confidence': float(det['confidence']),
                    'bbox': [int(coord) for coord in det['bbox']]
                }
                for det in detections
            ]
            
            # Publish detection data
            self.publisher.publish_detection(detection_data, frame_timestamp)
            
            # Publish metrics (example)
            metrics = {
                'fps': 0,  # Will be updated in the main loop
                'detection_count': len(detections),
                'timestamp': frame_timestamp
            }
            self.publisher.publish_metrics(metrics)
        
        return detections
    
    def run(self, source: str = '0', output: str = None) -> None:
        """Run the traffic monitoring application.
        
        Args:
            source: Video source (0 for webcam or path to video file)
            output: Output video file path (optional)
        """
        # Open video source
        if source.isdigit():
            source = int(source)
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer if output path is provided
        writer = None
        if output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output, fourcc, fps if fps > 0 else 30, (width, height))
        
        print("Starting traffic monitoring. Press 'q' to quit...")
        
        # Main loop
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break
                
                # Process frame
                detections = self.process_frame(frame)
                
                # Visualize detections
                frame_with_detections = self.detector.visualize_detections(frame.copy(), detections)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    fps = frame_count / (time.time() - start_time)
                    cv2.putText(frame_with_detections, f'FPS: {fps:.1f}', (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show the frame
                cv2.imshow('TrafficSense AI', frame_with_detections)
                
                # Write frame to output video
                if writer is not None:
                    writer.write(frame_with_detections)
                
                # Break the loop on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            # Clean up
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            
            # Disconnect from AWS IoT
            if self.mqtt_client:
                self.mqtt_client.disconnect()
            
            print("Traffic monitoring stopped.")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TrafficSense AI - Real-time traffic monitoring with AWS IoT')
    parser.add_argument('--model', type=str, default='models/ssd_mobilenet_v2_coco_quant_postprocess.tflite',
                      help='Path to the TFLite model')
    parser.add_argument('--source', type=str, default='0',
                      help='Video source (0 for webcam or path to video file)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output video file path (optional)')
    parser.add_argument('--no-iot', action='store_true',
                      help='Disable AWS IoT integration')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    # Create necessary directories
    os.makedirs('certs', exist_ok=True)
    
    # Initialize and run the traffic monitor
    monitor = TrafficMonitor(
        model_path=args.model,
        iot_enabled=not args.no_iot
    )
    
    monitor.run(
        source=args.source,
        output=args.output
    )
