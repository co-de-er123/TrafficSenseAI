""
Advanced detection and tracking module for TrafficSense AI.
Combines TensorFlow Lite for detection with OpenCV's tracking algorithms.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

class ObjectTrackerType(Enum):
    """Available object tracking algorithms in OpenCV."""
    CSRT = cv2.legacy.TrackerCSRT_create
    KCF = cv2.legacy.TrackerKCF_create
    MOSSE = cv2.legacy.TrackerMOSSE_create

@dataclass
class TrackedObject:
    """Represents a tracked object with its properties."""
    object_id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    class_id: int
    confidence: float
    tracker: Any
    last_updated: float
    consecutive_misses: int = 0
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float) -> None:
        """Update the object's properties."""
        self.bbox = bbox
        self.confidence = confidence
        self.last_updated = time.time()
        self.consecutive_misses = 0
    
    def miss(self) -> None:
        """Increment the miss counter for this object."""
        self.consecutive_misses += 1
        self.last_updated = time.time()

class AdvancedTrafficDetector:
    """
    Enhanced traffic detector with object tracking capabilities.
    Combines TensorFlow Lite for detection with OpenCV tracking.
    """
    
    def __init__(self, model_path: str, tracker_type: ObjectTrackerType = ObjectTrackerType.KCF,
                 detection_interval: int = 5, max_misses: int = 5):
        """
        Initialize the advanced traffic detector.
        
        Args:
            model_path: Path to the TFLite model
            tracker_type: Type of OpenCV tracker to use
            detection_interval: Run full detection every N frames
            max_misses: Maximum consecutive tracking failures before removing an object
        """
        # Base detector (from our previous implementation)
        from . import TrafficDetector
        self.detector = TrafficDetector(model_path)
        
        # Tracking parameters
        self.tracker_type = tracker_type
        self.detection_interval = detection_interval
        self.max_misses = max_misses
        self.frame_count = 0
        self.next_object_id = 0
        
        # Active trackers and objects
        self.tracked_objects: List[TrackedObject] = []
    
    def _create_tracker(self):
        """Create a new tracker instance."""
        return self.tracker_type.value()
    
    def _get_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        # Convert from (x1, y1, x2, y2) to (x, y, w, h)
        x1, y1, w1, h1 = box1[0], box1[1], box1[2]-box1[0], box1[3]-box1[1]
        x2, y2, w2, h2 = box2[0], box2[1], box2[2]-box2[0], box2[3]-box1[1]
        
        # Calculate intersection area
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        
        # Calculate union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _convert_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Convert from (x1, y1, x2, y2) to (x, y, w, h)."""
        return (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
    
    def _convert_bbox_back(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Convert from (x, y, w, h) to (x1, y1, x2, y2)."""
        return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
    
    def update(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Update trackers and perform detection if needed.
        
        Returns:
            List of detections with tracking IDs
        """
        self.frame_count += 1
        height, width = frame.shape[:2]
        
        # Run detection at specified interval or if we have no trackers
        if self.frame_count % self.detection_interval == 0 or not self.tracked_objects:
            # Get new detections
            detections = self.detector.detect(frame)
            
            # Convert detections to trackable format
            new_boxes = [self._convert_bbox(d['bbox']) for d in detections]
            
            # Match detections with existing trackers
            matched_det_indices = set()
            matched_tracker_indices = set()
            
            # For each existing tracker, find best matching detection
            for i, obj in enumerate(self.tracked_objects):
                if obj.consecutive_misses >= self.max_misses:
                    continue
                    
                best_iou = 0.3  # Minimum IOU to consider a match
                best_match = -1
                
                for j, box in enumerate(new_boxes):
                    if j in matched_det_indices:
                        continue
                        
                    iou = self._get_iou(
                        self._convert_bbox_back(obj.bbox),
                        self._convert_bbox_back(box)
                    )
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match = j
                
                if best_match != -1:
                    # Update tracker with new detection
                    obj.tracker = self._create_tracker()
                    obj.tracker.init(frame, new_boxes[best_match])
                    obj.update(
                        self._convert_bbox_back(new_boxes[best_match]),
                        detections[best_match]['confidence']
                    )
                    matched_det_indices.add(best_match)
                    matched_tracker_indices.add(i)
            
            # Add new detections as trackers
            for i, box in enumerate(new_boxes):
                if i not in matched_det_indices and detections[i]['confidence'] > 0.5:
                    tracker = self._create_tracker()
                    tracker.init(frame, box)
                    
                    obj = TrackedObject(
                        object_id=self.next_object_id,
                        bbox=self._convert_bbox_back(box),
                        class_id=detections[i]['class_id'],
                        confidence=detections[i]['confidence'],
                        tracker=tracker,
                        last_updated=time.time()
                    )
                    self.tracked_objects.append(obj)
                    self.next_object_id += 1
        
        # Update existing trackers
        current_time = time.time()
        valid_objects = []
        
        for obj in self.tracked_objects:
            if obj.consecutive_misses >= self.max_misses:
                continue
                
            success, box = obj.tracker.update(frame)
            
            if success:
                # Convert from (x, y, w, h) to (x1, y1, x2, y2)
                x1 = max(0, int(box[0]))
                y1 = max(0, int(box[1]))
                x2 = min(width, int(box[0] + box[2]))
                y2 = min(height, int(box[1] + box[3]))
                
                obj.bbox = (x1, y1, x2, y2)
                valid_objects.append(obj)
            else:
                obj.miss()
                if obj.consecutive_misses < self.max_misses:
                    valid_objects.append(obj)
        
        self.tracked_objects = valid_objects
        
        # Return current detections
        return [{
            'bbox': obj.bbox,
            'confidence': obj.confidence,
            'class_id': obj.class_id,
            'track_id': obj.object_id,
            'age': current_time - obj.last_updated
        } for obj in self.tracked_objects]
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detections on the frame with tracking information."""
        frame_copy = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            track_id = det.get('track_id', -1)
            confidence = det['confidence']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            if 'age' in det and det['age'] > 2.0:  # Older than 2 seconds
                color = (0, 165, 255)  # Orange
                
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with tracking ID and confidence
            label = f"ID:{track_id} {confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_copy, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame_copy
