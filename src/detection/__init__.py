"""
Object detection module for TrafficSense AI.
"""

from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any

class TrafficDetector:
    """Handles vehicle detection using TensorFlow Lite."""
    
    def __init__(self, model_path: str, threshold: float = 0.5):
        """Initialize the detector with a TensorFlow Lite model.
        
        Args:
            model_path: Path to the TFLite model file
            threshold: Confidence threshold for detections (0.0 to 1.0)
        """
        self.threshold = threshold
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Check input shape
        input_shape = self.input_details[0]['shape']
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess the input frame for the model."""
        # Resize and normalize the image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_width, self.input_height))
        input_data = np.expand_dims(img_resized.astype(np.float32), axis=0)
        input_data = (input_data - 127.5) / 127.5  # Normalize to [-1, 1]
        return input_data
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect vehicles in the input frame.
        
        Returns:
            List of detections, where each detection is a dictionary with keys:
            - 'bbox': [x1, y1, x2, y2] coordinates
            - 'confidence': Detection confidence score
            - 'class_id': Class ID of the detected object
        """
        # Preprocess the frame
        input_data = self.preprocess(frame)
        
        # Set the input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensors
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        # Process detections
        detections = []
        height, width = frame.shape[:2]
        
        for i in range(len(scores)):
            if scores[i] > self.threshold:
                # Scale bounding box coordinates to the original frame size
                y1, x1, y2, x2 = boxes[i]
                x1 = int(x1 * width)
                y1 = int(y1 * height)
                x2 = int(x2 * width)
                y2 = int(y2 * height)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(scores[i]),
                    'class_id': int(classes[i])
                })
        
        return detections
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detections on the frame."""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_id = det['class_id']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f'Vehicle: {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
