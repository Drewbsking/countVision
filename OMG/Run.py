import cv2
from ultralytics import YOLO

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
crossing_line = 320  # Define the vertical line x-coordinate
crossed_objects = set()
crossing_count = 0
previous_positions = {}

def is_crossing_right_to_left(prev_x, curr_x, line_x):
    return prev_x > line_x and curr_x < line_x

while processed_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to retrieve frame.")
        break

    # Skip frames to lower the frame rate
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    # Optionally resize frame to improve performance
    frame = cv2.resize(frame, (640, 480))

    # Perform inference using YOLO model
    results = model(frame)

    # Draw bounding boxes on the frame and provide status message
    detections_count = 0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            # Assuming class IDs for cars, buses, and trucks are [2, 5, 7] (you can adjust these)
            if class_id not in [2, 5, 7]:
                continue  # Skip other classes

            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            label = f'{model.names[class_id]}: {confidence:.2f}'
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            detections_count += 1

            # Calculate the midpoint of the bounding box
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            # Use the midpoint as a unique key
            object_id = (mid_x, mid_y)

            # Check if the object is crossing the line from right to left
            if object_id in previous_positions:
                prev_x = previous_positions[object_id]
                if is_crossing_right_to_left(prev_x, mid_x, crossing_line):
                    if object_id not in crossed_objects:
                        crossed_objects.add(object_id)
                        crossing_count += 1
                        print(f"Car crossed the line. Total crossings: {crossing_count}")
                    else:
                        print(f"Object {object_id} already counted.")
                else:
                    print(f"Object {object_id} did not cross from right to left. Previous X: {prev_x}, Current X: {mid_x}")
            else:
                print(f"New object detected at position: {mid_x}, {mid_y}")

            # Update the previous position
            previous_positions[object_id] = mid_x

    print(f"Frame {processed_count}: {detections_count} relevant objects detected")

    # Draw the crossing line
    cv2.line(frame, (crossing_line, 0), (crossing_line, 480), (0, 0, 255), 2)

    # Display the running total of car crossings
    cv2.putText(frame, f'Total crossings: {crossing_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
print(f"Total cars crossed the line: {crossing_count}")
