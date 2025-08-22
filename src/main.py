"""
TrafficSense AI - Main application entry point.
"""

import os
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from detection import TrafficDetector

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TrafficSense AI - Real-time traffic monitoring')
    parser.add_argument('--model', type=str, default='models/ssd_mobilenet_v2_coco_quant_postprocess.tflite',
                       help='Path to the TFLite model')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold (0.0 to 1.0)')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source (0 for webcam or path to video file)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path (optional)')
    return parser.parse_args()

def main():
    """Main function to run the traffic monitoring system."""
    args = parse_arguments()
    
    # Initialize the detector
    print(f"Loading model from {args.model}...")
    detector = TrafficDetector(args.model, args.threshold)
    
    # Open video source
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path is provided
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps if fps > 0 else 30, (width, height))
    
    print("Starting traffic monitoring. Press 'q' to quit...")
    
    # Main loop
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break
        
        # Perform detection
        detections = detector.detect(frame)
        
        # Visualize detections
        frame_with_detections = detector.visualize_detections(frame.copy(), detections)
        
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
    
    # Clean up
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    
    print("Traffic monitoring stopped.")

if __name__ == '__main__':
    main()
