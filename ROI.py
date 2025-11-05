import cv2
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict

# Centroid tracker class to keep track of objects
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids), axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects


# Define the ROI points
ROI_POINTS = np.array([(360, 745), (519, 867), (672, 774), (504, 664)], np.int32)
ROI_POINTS = ROI_POINTS.reshape((-1, 1, 2))

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Initialize video capture
cap = cv2.VideoCapture("2024-10-02 07-46-07-1.mp4")

# Initialize the centroid tracker and counter
tracker = CentroidTracker()
car_counter = 0
exited_objects = set()

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Perform inference on the frame using YOLOv8
    results = model(frame)

    # Draw the defined ROI in Red
    cv2.polylines(frame, [ROI_POINTS], isClosed=True, color=(0, 0, 255), thickness=3)

    new_centroids = []

    # Iterate through detected objects
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])

            if class_id == 2 and confidence > 0.5:  # Class ID 2 represents cars
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                centroid = (centroid_x, centroid_y)

                # Check if the centroid is inside the ROI
                if cv2.pointPolygonTest(ROI_POINTS, (centroid_x, centroid_y), False) >= 0:
                    new_centroids.append(centroid)

                    # Draw bounding box and centroid on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

    # Update the tracker with new centroids
    objects = tracker.update(new_centroids)

    # Draw the tracked object IDs and count cars exiting the ROI
    for (object_id, centroid) in objects.items():
        text = f"ID {object_id}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)

        # Check if the object has exited the ROI
        if object_id not in exited_objects and not is_in_roi(centroid, ROI_POINTS):
            car_counter += 1
            exited_objects.add(object_id)

    # Display the car counter on the video frame
    cv2.putText(frame, f"Car Count: {car_counter}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Show the frame
    cv2.imshow("Car Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
