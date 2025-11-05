import cv2
import torch

# Define the RTSP stream URL
rtsp_url = "rtsp://10.32.162.146:8554/swVideo1"

# Initialize video capture object for the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the stream opened successfully
if not cap.isOpened():
    print(f"Error: Unable to connect to the RTSP stream at {rtsp_url}")
    exit()

# Load the YOLOv5 model (smallest model, adjust as needed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Process the video stream frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from the stream.")
        break
    fps_stream = cap.get(cv2.CAP_PROP_FPS)
    print(f"Stream FPS: {fps_stream}")
    
    # Perform vehicle detection on the frame
    results = model(frame)

    # Render the results on the frame
    frame = results.render()[0]

    # Display the frame with detected vehicles
    cv2.imshow("Vehicle Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
