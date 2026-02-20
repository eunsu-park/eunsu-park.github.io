"""
Real-time Object Tracking with OpenCV

This script demonstrates multiple object tracking algorithms:
- KCF (Kernelized Correlation Filters)
- CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)
- MOSSE (Minimum Output Sum of Squared Error)
- Centroid-based tracking (from scratch)
- Kalman filter for motion prediction

Requirements:
    pip install opencv-contrib-python numpy

Author: Claude
Date: 2026-02-15
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import sys


class CentroidTracker:
    """
    Simple centroid-based object tracker.

    Tracks objects by computing centroids and matching them across frames
    using Euclidean distance.
    """

    def __init__(self, max_disappeared: int = 50):
        """
        Args:
            max_disappeared: Maximum frames an object can disappear before removal
        """
        self.next_object_id = 0
        self.objects: Dict[int, Tuple[int, int]] = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid: Tuple[int, int]) -> int:
        """Register a new object with its centroid."""
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id: int):
        """Remove an object from tracking."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int]]:
        """
        Update tracker with new detections.

        Args:
            rects: List of bounding boxes (x, y, w, h)

        Returns:
            Dictionary mapping object IDs to centroids
        """
        # If no detections, mark all objects as disappeared
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Compute centroids of new detections
        input_centroids = []
        for (x, y, w, h) in rects:
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            input_centroids.append((cx, cy))

        # If no existing objects, register all new centroids
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # Match existing objects to new centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distance matrix
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, input_centroid in enumerate(input_centroids):
                    D[i, j] = np.linalg.norm(
                        np.array(obj_centroid) - np.array(input_centroid)
                    )

            # Find minimum distance assignments
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            # Update matched objects
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched objects
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new objects
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


class KalmanTracker:
    """Kalman filter-based tracker for motion prediction."""

    def __init__(self):
        """Initialize Kalman filter for 2D position tracking."""
        # State: [x, y, vx, vy] (position and velocity)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

    def predict(self) -> Tuple[int, int]:
        """Predict next position."""
        prediction = self.kalman.predict()
        return int(prediction[0]), int(prediction[1])

    def update(self, x: int, y: int) -> Tuple[int, int]:
        """Update with measurement and return corrected position."""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(measurement)
        return x, y


def compute_iou(box1: Tuple[int, int, int, int],
                box2: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1, box2: Bounding boxes in format (x, y, w, h)

    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Compute intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Compute union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def generate_synthetic_video(output_path: str,
                            num_frames: int = 300,
                            width: int = 640,
                            height: int = 480) -> str:
    """
    Generate a synthetic video with moving objects for testing.

    Args:
        output_path: Path to save the video
        num_frames: Number of frames to generate
        width, height: Video dimensions

    Returns:
        Path to the generated video
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    # Define moving objects (x, y, vx, vy, size, color)
    objects = [
        [50, 50, 2, 1, 30, (0, 255, 0)],      # Green square
        [300, 100, -1, 2, 25, (255, 0, 0)],    # Blue square
        [100, 300, 1.5, -1, 35, (0, 0, 255)],  # Red square
    ]

    for frame_idx in range(num_frames):
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Update and draw objects
        for obj in objects:
            x, y, vx, vy, size, color = obj

            # Draw object
            cv2.rectangle(frame,
                         (int(x), int(y)),
                         (int(x + size), int(y + size)),
                         color, -1)

            # Update position
            obj[0] += vx
            obj[1] += vy

            # Bounce off walls
            if obj[0] <= 0 or obj[0] >= width - size:
                obj[2] *= -1
            if obj[1] <= 0 or obj[1] >= height - size:
                obj[3] *= -1

        # Add frame number
        cv2.putText(frame, f'Frame: {frame_idx}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"Generated synthetic video: {output_path}")
    return output_path


def demo_opencv_trackers(video_path: Optional[str] = None):
    """
    Demonstrate OpenCV's built-in tracking algorithms.

    Args:
        video_path: Path to input video (None for synthetic video)
    """
    # Generate synthetic video if no input provided
    if video_path is None:
        video_path = '/tmp/tracking_demo.avi'
        generate_synthetic_video(video_path)

    # Available tracker types
    tracker_types = {
        'KCF': cv2.TrackerKCF_create,
        'CSRT': cv2.TrackerCSRT_create,
    }

    # Try to add MOSSE tracker if available (in legacy module)
    try:
        tracker_types['MOSSE'] = cv2.legacy.TrackerMOSSE_create
    except AttributeError:
        print("MOSSE tracker not available (requires opencv-contrib-python)")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        return

    # Select ROI for tracking
    print("Select ROI to track, then press SPACE or ENTER")
    bbox = cv2.selectROI("Select ROI", frame, False)
    cv2.destroyWindow("Select ROI")

    if bbox[2] == 0 or bbox[3] == 0:
        print("No ROI selected, using default")
        bbox = (100, 100, 60, 60)

    # Initialize trackers
    trackers = {}
    for name, create_fn in tracker_types.items():
        tracker = create_fn()
        tracker.init(frame, bbox)
        trackers[name] = tracker

    # Initialize Kalman filter
    kalman = KalmanTracker()
    cx, cy = int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)
    kalman.update(cx, cy)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display = frame.copy()

        # Track with each algorithm
        y_offset = 30
        for name, tracker in trackers.items():
            success, box = tracker.update(frame)

            if success:
                x, y, w, h = [int(v) for v in box]
                color = (0, 255, 0) if name == 'CSRT' else (255, 0, 0) if name == 'KCF' else (0, 165, 255)
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display, f'{name}', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Update Kalman filter with CSRT results (most accurate)
                if name == 'CSRT':
                    cx, cy = int(x + w/2), int(y + h/2)
                    kalman.update(cx, cy)
            else:
                cv2.putText(display, f'{name}: Lost', (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            y_offset += 25

        # Show Kalman prediction
        pred_x, pred_y = kalman.predict()
        cv2.circle(display, (pred_x, pred_y), 5, (255, 255, 0), -1)
        cv2.putText(display, 'Kalman', (pred_x+10, pred_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.putText(display, f'Frame: {frame_count}', (10, frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Object Tracking', display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)  # Pause

    cap.release()
    cv2.destroyAllWindows()


def demo_centroid_tracking():
    """Demonstrate centroid-based tracking with background subtraction."""
    # Generate synthetic video
    video_path = '/tmp/centroid_tracking_demo.avi'
    generate_synthetic_video(video_path, num_frames=200)

    cap = cv2.VideoCapture(video_path)
    tracker = CentroidTracker(max_disappeared=30)

    # Background subtractor for detecting moving objects
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=100, varThreshold=40, detectShadows=True
    )

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)

        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Extract bounding boxes
        rects = []
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Filter small objects
                continue
            x, y, w, h = cv2.boundingRect(contour)
            rects.append((x, y, w, h))

        # Update tracker
        objects = tracker.update(rects)

        # Draw tracked objects
        for object_id, centroid in objects.items():
            color = colors[object_id % len(colors)]
            cv2.circle(frame, centroid, 5, color, -1)
            cv2.putText(frame, f'ID: {object_id}',
                       (centroid[0] - 10, centroid[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        cv2.putText(frame, f'Frame: {frame_count} | Objects: {len(objects)}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Centroid Tracking', frame)
        cv2.imshow('Foreground Mask', fg_mask)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


def demo_multi_tracker():
    """Demonstrate tracking multiple objects simultaneously."""
    # Generate synthetic video
    video_path = '/tmp/multi_tracking_demo.avi'
    generate_synthetic_video(video_path, num_frames=200)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        print("Error: Cannot read video")
        return

    # Create MultiTracker
    multi_tracker = cv2.legacy.MultiTracker_create()

    # Define multiple ROIs
    bboxes = [
        (50, 50, 40, 40),
        (300, 100, 35, 35),
        (100, 300, 45, 45)
    ]

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

    # Initialize trackers
    for bbox in bboxes:
        tracker = cv2.TrackerCSRT_create()
        multi_tracker.add(tracker, frame, bbox)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Update all trackers
        success, boxes = multi_tracker.update(frame)

        # Draw tracked objects
        for i, box in enumerate(boxes):
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x+w, y+h), colors[i], 2)
            cv2.putText(frame, f'Object {i+1}', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

        cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Multi-Object Tracking', frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Object Tracking Demonstrations")
    print("=" * 50)
    print("\nAvailable demos:")
    print("1. OpenCV Trackers (KCF, CSRT, MOSSE) + Kalman Filter")
    print("2. Centroid-Based Tracking")
    print("3. Multi-Object Tracking")
    print("\nPress Ctrl+C to exit anytime")

    try:
        # Demo 1: OpenCV trackers
        print("\n[Demo 1] OpenCV Trackers with Kalman Filter")
        print("-" * 50)
        demo_opencv_trackers()

        # Demo 2: Centroid tracking
        print("\n[Demo 2] Centroid-Based Tracking")
        print("-" * 50)
        demo_centroid_tracking()

        # Demo 3: Multi-tracker
        print("\n[Demo 3] Multi-Object Tracking")
        print("-" * 50)
        demo_multi_tracker()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    cv2.destroyAllWindows()
    print("\nAll demos completed!")
