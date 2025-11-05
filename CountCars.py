import cv2
import numpy as np
import time

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video capture
cap = cv2.VideoCapture("1900-151662242_small.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

print("Processing video...")

vehicle_counts = {"car": 0, "truck": 0, "motorcycle": 0}
frame_count = 0

# Start time
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Finished reading video.")
        break

    frame_count += 1

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # NMS threshold

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label in vehicle_counts:
                vehicle_counts[label] += 1
                print(f"Detected {label} at frame {frame_count}, bbox: [{x}, {y}, {w}, {h}]")

cap.release()
cv2.destroyAllWindows()

# End time
end_time = time.time()
processing_time = end_time - start_time

print("Vehicle counts:", vehicle_counts)
print(f"Processing time: {processing_time:.2f} seconds")
