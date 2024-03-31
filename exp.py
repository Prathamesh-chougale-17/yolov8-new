# import cv2
from ultralytics import YOLO


# Load YOLOv8 model
model = YOLO("posturedetection.pt")

# IP Webcam stream URL
# url = "http://192.168.0.106:8080/video"
# url = "http://192.168.0.105:8080/video"

# Open video stream
results = model.predict(source=0, show=True, stream=True)
for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs

print(results)
