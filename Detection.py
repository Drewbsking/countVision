import cv2
from ultralytics import YOLO

# Replace with your RTSP URL
rtsp_url = 'rtsp://10.32.161.62:8554/swVideo'

# Initialize video capture from the RTSP stream with increased timeout settings
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Set buffer size
cap.set(cv2.CAP_PROP_POS_MSEC, 30000)  # Increase timeout to 30 seconds

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
else:
    print("RTSP stream opened successfully.")

# Load YOLO model (ensure the model path is correct or use a pre-trained model)
model = YOLO('yolov8s.pt')  # Using YOLOv8s model

frame_count = 0
processed_count = 0
max_frames = 1000  # Set the maximum number of frames to process
frame_skip = 10  # Process every 10th frame
skip_class_ids = [9]  # Assuming the class ID for traffic lights is 9

while processed_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to retrieve frame. Attempting to reconnect...")
        cap.release()
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Set buffer size
        cap.set(cv2.CAP_PROP_POS_MSEC, 30000)  # Increase timeout to 30 seconds
        if not cap.isOpened():
            print("Error: Could not reopen RTSP stream.")
            break
        continue

    # Skip frames to lower the frame rate
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    # Optionally resize frame to improve performance
    frame = cv2.resize(frame, (640, 480))

    # Perform inference using YOLO model
    results = model(frame)

    # Draw bounding boxes on the frame
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            if class_id in skip_class_ids:
                continue  # Skip traffic lights or other unwanted classes

            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            label = f'{model.names[class_id]}: {confidence:.2f}'
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('RTSP Stream with Detections', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1
    processed_count += 1

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

print(f"Processed {processed_count} frames.")
