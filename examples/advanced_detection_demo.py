""
Advanced detection and tracking demo for TrafficSense AI.
Shows how to use the AdvancedTrafficDetector with object tracking.
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.detection.advanced import AdvancedTrafficDetector, ObjectTrackerType

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Advanced Traffic Detection Demo')
    parser.add_argument('--model', type=str, 
                      default='models/ssd_mobilenet_v2_coco_quant_postprocess.tflite',
                      help='Path to the TFLite model')
    parser.add_argument('--source', type=str, default='0',
                      help='Video source (0 for webcam or path to video file)')
    parser.add_argument('--tracker', type=str, default='KCF',
                      choices=['CSRT', 'KCF', 'MOSSE'],
                      help='Tracker type to use')
    parser.add_argument('--detection-interval', type=int, default=5,
                      help='Run full detection every N frames')
    parser.add_argument('--max-misses', type=int, default=5,
                      help='Maximum consecutive tracking failures before removing an object')
    parser.add_argument('--output', type=str, default=None,
                      help='Output video file path (optional)')
    return parser.parse_args()

def main():
    """Run the advanced detection demo."""
    args = parse_arguments()
    
    # Map tracker type string to enum
    tracker_type = getattr(ObjectTrackerType, args.tracker.upper())
    
    # Initialize the detector
    print(f"Initializing detector with {args.tracker} tracker...")
    detector = AdvancedTrafficDetector(
        model_path=args.model,
        tracker_type=tracker_type,
        detection_interval=args.detection_interval,
        max_misses=args.max_misses
    )
    
    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
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
    
    print("Starting advanced detection demo. Press 'q' to quit...")
    
    # Main loop
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            # Run detection and tracking
            detections = detector.update(frame)
            
            # Draw detections
            frame_with_detections = detector.draw_detections(frame, detections)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 10 == 0:
                current_fps = frame_count / (time.time() - start_time)
                cv2.putText(frame_with_detections, f'FPS: {current_fps:.1f}', (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display tracker info
            cv2.putText(frame_with_detections, 
                       f'Tracker: {args.tracker} | Objects: {len(detections)}', 
                       (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow('Advanced Traffic Detection', frame_with_detections)
            
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
        print("Demo finished.")

if __name__ == '__main__':
    main()
