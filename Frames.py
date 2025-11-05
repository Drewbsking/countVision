import cv2
import os

# Replace with your RTSP URL
#rtsp_url = 'rtmp://192.168.13.92:1935/rtplive/RCOC_285'
rtsp_url = 'C:\\Users\\abates\\Videos\\2024-07-18 10-36-27.mkv'
# Initialize video capture from the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
else:
    print("RTSP stream opened successfully.")

frame_count = 0
save_dir = "frames"

# Create directory to save frames if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to retrieve frame.")
        break

    # Save the frame to disk
    frame_path = os.path.join(save_dir, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_path, frame)
    print(f"Saved frame {frame_count} to {frame_path}")
    frame_count += 1

    # Break the loop after saving a certain number of frames
    if frame_count >= 10:  # Save only the first 10 frames for example
        break

# Release the capture
cap.release()
print("Finished saving frames.")
