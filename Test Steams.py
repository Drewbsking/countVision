import cv2

def test_stream(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Error: Could not open stream {url}")
    else:
        print(f"Stream {url} opened successfully.")
        ret, frame = cap.read()
        if ret:
            print(f"Retrieved frame from {url} successfully.")
        else:
            print(f"Error: Failed to retrieve frame from {url}.")
    cap.release()

rtsp_url = 'rtsp://10.32.161.62:8554/swVideo'
rtmp_url = 'rtmp://192.168.13.92:1935/rtplive/RCOC_285'

test_stream(rtsp_url)
test_stream(rtmp_url)
