import cv2
import numpy as np
import time
from scipy.spatial import distance as dist
from imutils.video import FPS

# Paths to YOLOv4-tiny files
model_cfg = 'yolov4-tiny.cfg'
model_weights = 'yolov4-tiny.weights'
class_file = 'coco.names'

# Load class names
with open(class_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLOv4-tiny model
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
# Enable GPU support if you have a CUDA-compatible GPU
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Color setup for bounding boxes
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Define the RTSP stream URL
rtsp_url = "rtsp://10.32.162.146:8554/swVideo1"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print(f"Error: Could not open the stream at {rtsp_url}")
    exit()

# Centroid Tracker Class for Tracking IDs
class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
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

# Initialize Centroid Tracker and FPS Counter
tracker = CentroidTracker(max_disappeared=20, max_distance=60)
fps = FPS().start()
frame_count = 0  # Counter to keep track of frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Retrying...")
        break

    # Reduce the frame size to 640x360 for faster processing
    frame = cv2.resize(frame, (640, 360))
    h, w = frame.shape[:2]

    frame_count += 1
    if frame_count % 5 == 0:  # Process every 5th frame to reduce CPU load
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id] in ["car", "truck", "bus", "motorbike"]:
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append((x, y, x + width, y + height))

        # Update tracker with bounding boxes
        objects = tracker.update(boxes)
    else:
        # Just update tracker without new detections
        objects = tracker.update([])

    # Draw bounding boxes and IDs on the frame
    for (object_id, centroid) in objects.items():
        text = f"ID {object_id}"
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    cv2.imshow("Vehicle Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    fps.update()

# Stop the FPS counter
fps.stop()
print(f"Elapsed time: {fps.elapsed():.2f}")
print(f"Approx. FPS: {fps.fps():.2f}")

cap.release()
cv2.destroyAllWindows()
