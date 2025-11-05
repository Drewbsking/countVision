import cv2
import numpy as np

# Points defining the ROI: (360, 745), (519, 867), (672, 774), (504, 664)
ROI_POINTS = np.array([(360, 745), (519, 867), (672, 774), (504, 664)], np.int32)
ROI_POINTS = ROI_POINTS.reshape((-1, 1, 2))

# Initialize video capture
cap = cv2.VideoCapture("2024-10-02 07-46-07-1.mp4")

# Create background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

# Variables for tracking and counting
car_counter = 0
min_contour_width = 40  # Minimum width of detected car
min_contour_height = 40  # Minimum height of detected car
line_y_position = 720    # Line position for counting cars

# Define a list to keep track of vehicles' centroids
tracked_centroids = []

def is_in_roi(centroid, roi_points):
    """ Check if a point is inside the defined ROI. """
    return cv2.pointPolygonTest(roi_points, centroid, False) >= 0

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and apply background subtraction
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = bg_subtractor.apply(gray_frame)

    # Apply morphological operations to filter noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Detect contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the defined ROI
    cv2.polylines(frame, [ROI_POINTS], isClosed=True, color=(0, 255, 255), thickness=2)

    # Iterate through contours and filter based on size
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue  # Skip small contours to filter noise

        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_contour_width and h >= min_contour_height:
            # Calculate the centroid of the contour
            centroid_x = int(x + w / 2)
            centroid_y = int(y + h / 2)
            centroid = (centroid_x, centroid_y)

            # Check if the centroid is inside the ROI
            if is_in_roi(centroid, ROI_POINTS):
                tracked_centroids.append(centroid)

                # Draw the bounding box around the car
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box

                # Draw a circle at the centroid
                cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Red dot for centroid

                # Check if the car is crossing the counting line
                if line_y_position - 5 <= centroid_y <= line_y_position + 5:
                    car_counter += 1

    # Display the car counter in the upper-left corner
    cv2.putText(frame, f"Car Count: {car_counter}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Show the frame with detection and counting
    cv2.imshow("Car Counting", frame)

    # Exit on pressing the `q` key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
