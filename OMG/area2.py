import cv2
from ultralytics import YOLO
import numpy as np
import sys
import os

# Add the directory containing sort.py to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sort'))

from sort import Sort

# Replace with your video file path
video_file_path = 'C:\\Users\\abates\\Videos\\2024-07-18 10-36-27.mkv'

# Initialize video capture from the local video file
cap = cv2.VideoCapture(video_file_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    print("Video file opened successfully.")

# Load YOLO model (ensure the model path is correct or use a pre-trained model)
model = YOLO('yolov8s.pt')  # Using YOLOv8s model

frame_count = 0
processed_count = 0
max_frames = 1000  # Set the maximum number of frames to process
frame_skip = 10  # Process every 10th frame

# Define the area (polygon) for counting vehicles
polygon = np.array([[950, 440], [1410, 480], [1311, 560], [745, 540]])

# Initialize SORT tracker
tracker = Sort()

# Function to check if a point is inside a polygon
def is_inside_polygon(x, y, poly):
    return cv2.pointPolygonTest(poly, (x, y), False) >= 0

crossing_count = 0
tracked_objects = {}  # Dictionary to keep track of counted objects

while processed_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to retrieve frame.")
        break

    # Skip frames to lower the frame rate
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    # Perform inference using YOLO model
    results = model(frame)

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            # Assuming class IDs for cars, buses, and trucks are [2, 5, 7] (you can adjust these)
            if class_id not in [2, 5, 7]:
                continue  # Skip other classes

            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            detections.append([x1, y1, x2, y2, confidence])

    # Update tracker with current frame detections
    tracked_objects_array = tracker.update(np.array(detections))

    # Draw bounding boxes and track IDs on the frame
    for obj in tracked_objects_array:
        x1, y1, x2, y2, obj_id = map(int, obj)
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check if the midpoint is inside the polygon
        if is_inside_polygon(mid_x, mid_y, polygon):
            if obj_id not in tracked_objects:
                tracked_objects[obj_id] = True  # Mark object as tracked
                crossing_count += 1
                print(f"Vehicle ID {obj_id} entered the area. Total count: {crossing_count}")

    print(f"Frame {processed_count}: {len(tracked_objects_array)} relevant objects tracked")

    # Draw the defined area (polygon)
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)

    # Display the running total of vehicle entries
    cv2.putText(frame, f'Total count: {crossing_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame with detections
    cv2.imshow('Video with Detections', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1
    processed_count += 1

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

print(f"Processed {processed_count} frames.")
print(f"Total vehicles entered the area: {crossing_count}")
