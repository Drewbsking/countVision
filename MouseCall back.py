import cv2
import numpy as np

# Global variable to store coordinates
points = []

# Mouse callback function
def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        points.append((x, y))  # Store the point
        print(f"Point selected: ({x}, {y})")

# Load a sample frame from your video to select ROI
cap = cv2.VideoCapture(r"P:\My Drive\_TS\Code\CountCars\2024-10-02 07-46-07-1.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from video.")
    cap.release()
    exit()

# Resize the frame
frame_resized = cv2.resize(frame, (480, 320))

# Display frame and set up mouse callback
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_points)

while True:
    # Show the resized frame
    cv2.imshow("Frame", frame_resized)

    # Press 'q' to quit and finalize ROI selection
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Display the selected points and draw the ROI
print(f"Selected points: {points}")

# Draw the selected ROI on the resized frame (optional)
if len(points) > 1:
    # Convert the points to a format suitable for cv2.polylines
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame_resized, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imshow("ROI", frame_resized)
    cv2.waitKey(0)

# Release resources
cap.release()
cv2.destroyAllWindows()
