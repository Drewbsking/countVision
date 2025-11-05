import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from collections import deque

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the YOLOv5s model using the ultralytics package
model = YOLO('yolov5s.pt').to(device)

# Get class IDs for 'car' and 'truck'
car_class_id = None
truck_class_id = None
for cls_id, cls_name in model.names.items():
    if cls_name == 'car':
        car_class_id = cls_id
    elif cls_name == 'truck':
        truck_class_id = cls_id

if car_class_id is None or truck_class_id is None:
    print("Error: Could not find 'car' or 'truck' classes in model.names")
    exit()

# List of class IDs to detect
target_class_ids = [car_class_id, truck_class_id]

# Define the RTSP stream URL
rtsp_url = "rtsp://10.32.162.146:8554/swVideo"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print(f"Error: Could not open the stream at {rtsp_url}")
    exit()

cv2.namedWindow('RTSP Stream - YOLO Detection', cv2.WINDOW_NORMAL)

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()
fps = 0

# Frame skipping variables
frame_counter = 0
frame_skip = 2  # Process every 2nd frame

# Variables to store last detections
last_detections = []

# Initialize the tracker
tracker = {}  # Dictionary to hold tracking information
track_id = 0  # Initial track ID
track_history = {}  # To store the history of centroids
max_history = 30  # Number of frames to keep in history

# Vehicle counting variables
vehicle_count = 0
counted_ids = set()

# Define the line (coordinates of two points)
line_position = ((148, 196), (199, 234))  # Adjust these coordinates as needed

def intersect(A, B, C, D):
    # Return True if line segments AB and CD intersect
    def ccw(X, Y, Z):
        return (Z[1] - X[1]) * (Y[0] - X[0]) > (Y[1] - X[1]) * (Z[0] - X[0])

    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_counter += 1

    # Resize the frame to 480x320 for consistent display
    frame_resized = cv2.resize(frame, (480, 320))

    # Get frame dimensions
    height, width = frame_resized.shape[:2]
    resolution = f"{width}x{height}"

    # Draw the counting line
    cv2.line(frame_resized, line_position[0], line_position[1], (0, 0, 255), 2)

    # If the frame is skipped, use last detections
    if frame_counter % frame_skip != 0:
        # Update tracking history
        for obj_id in list(tracker.keys()):
            tracker[obj_id]['age'] += 1
            if tracker[obj_id]['age'] > max_history:
                del tracker[obj_id]
                del track_history[obj_id]
        # Draw last detections and tracks
        for obj_id, info in tracker.items():
            centroid = info['centroid']
            x1, y1, x2, y2 = info['bbox']
            class_name = info['class_name']
            label = f"{class_name.capitalize()} ID {obj_id}"
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame_resized, centroid, 4, (255, 0, 0), -1)

        # Display vehicle count
        count_text = f"Vehicle Count: {vehicle_count}"
        cv2.putText(frame_resized, count_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Prepare overlay text
        overlay_text = f"FPS: {fps:.2f} | Resolution: {resolution}"

        # Put overlay text on the frame
        cv2.putText(frame_resized, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Display the frame
        cv2.imshow('RTSP Stream - YOLO Detection', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break
        continue  # Skip processing this frame

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Perform object detection, specifying target classes
    with torch.no_grad():
        results = model(frame_rgb, classes=target_class_ids)

    detections = []
    # Process results
    for result in results:
        boxes = result.boxes  # Boxes object
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Only process detections for 'car' and 'truck'
            if class_id in target_class_ids:
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                confidence = float(confidence)
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'centroid': centroid,
                    'class_name': class_name,
                    'confidence': confidence
                }
                detections.append(detection)

    # Simple tracking logic
    new_tracker = {}
    for det in detections:
        centroid = det['centroid']
        bbox = det['bbox']
        class_name = det['class_name']
        min_dist = float('inf')
        min_id = None

        # Find the closest existing tracked object
        for obj_id, info in tracker.items():
            prev_centroid = info['centroid']
            dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
            if dist < min_dist and dist < 50:  # Distance threshold
                min_dist = dist
                min_id = obj_id

        if min_id is not None:
            # Update existing object
            new_tracker[min_id] = {
                'centroid': centroid,
                'bbox': bbox,
                'class_name': class_name,
                'age': 0
            }
            track_history[min_id].append(centroid)
            if len(track_history[min_id]) > max_history:
                track_history[min_id].popleft()
        else:
            # Assign new ID
            track_id += 1
            new_tracker[track_id] = {
                'centroid': centroid,
                'bbox': bbox,
                'class_name': class_name,
                'age': 0
            }
            track_history[track_id] = deque([centroid], maxlen=max_history)

    tracker = new_tracker

    # Check for line crossing
    for obj_id, info in tracker.items():
        if obj_id in counted_ids:
            continue
        history = track_history[obj_id]
        if len(history) >= 2:
            p0 = history[-2]
            p1 = history[-1]
            if intersect(p0, p1, line_position[0], line_position[1]):
                vehicle_count += 1
                counted_ids.add(obj_id)
                print(f"Vehicle ID {obj_id} crossed the line.")

    # Draw detections and tracks
    for obj_id, info in tracker.items():
        centroid = info['centroid']
        x1, y1, x2, y2 = info['bbox']
        class_name = info['class_name']
        label = f"{class_name.capitalize()} ID {obj_id}"
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_resized, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame_resized, centroid, 4, (255, 0, 0), -1)

    # Display vehicle count
    count_text = f"Vehicle Count: {vehicle_count}"
    cv2.putText(frame_resized, count_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Prepare overlay text
    overlay_text = f"FPS: {fps:.2f} | Resolution: {resolution}"

    # Put overlay text on the frame
    cv2.putText(frame_resized, overlay_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('RTSP Stream - YOLO Detection', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
