# import cv2
# import numpy as np
# import urllib.request

# # Load YOLOv8 model
# net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# # Set input and output layers
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# # Function to perform object detection
# def detect_objects(frame):
#     # Preprocess the frame
#     blob = cv2.dnn.blobFromImage(
#         frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
#     )
#     net.setInput(blob)
#     outs = net.forward(output_layers)

#     # Get bounding boxes, confidences, and class IDs
#     class_ids = []
#     confidences = []
#     boxes = []
#     height, width, _ = frame.shape
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # Apply non-maximum suppression to remove overlapping bounding boxes
#     # Define the classes
#     classes = ["class1", "class2", "class3"]

#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#     # Draw bounding boxes and labels on the frame
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(
#                 frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
#             )

#     return frame


# # URL of the video to defog
# video_url = "https://example.com/video.mp4"

# # Open the video stream
# stream = urllib.request.urlopen(video_url)
# bytes = bytes()

# # Read and process each frame of the video
# while True:
#     bytes += stream.read(1024)
#     a = bytes.find(b"\xff\xd8")
#     b = bytes.find(b"\xff\xd9")
#     if a != -1 and b != -1:
#         jpg = bytes[a : b + 2]
#         bytes = bytes[b + 2 :]
#         frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

#         # Perform defogging on the frame (add your defogging code here)


#         # Perform object detection on the defogged frame
#         frame = detect_objects(frame)

#         # Display the frame
#         cv2.imshow("Defogged Video", frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

# # Release the video stream and close all windows
# stream.release()
# cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2
import numpy as np

url = "https://next-practise-portfolio.vercel.app/expv.mp4"
cap = cv2.VideoCapture(url)
model = YOLO("yolov8l.pt")


def defog_dcp(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_channel = cv2.min(gray, axis=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    transmission = cv2.max(dark_channel, 0.1)
    transmission = cv2.erode(transmission, kernel)

    atmospheric_light = cv2.min(frame, axis=2)
    atmospheric_light = cv2.max(atmospheric_light, axis=0)

    dehazed_frame = np.zeros_like(frame)
    for i in range(3):
        dehazed_frame[:, :, i] = (
            frame[:, :, i] - atmospheric_light[i]
        ) / transmission + atmospheric_light[i]

    dehazed_frame = np.clip(dehazed_frame, 0, 1)  # Ensure values are within 0-1 range
    return dehazed_frame


def defog_nld(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_channel = cv2.min(gray, axis=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    transmission = cv2.max(dark_channel, 0.1)
    transmission = cv2.erode(transmission, kernel)

    atmospheric_light = cv2.min(frame, axis=2)
    atmospheric_light = cv2.max(atmospheric_light, axis=0)

    dehazed_frame = frame - atmospheric_light
    dehazed_frame = dehazed_frame / transmission + atmospheric_light

    dehazed_frame = np.clip(dehazed_frame, 0, 1)
    return dehazed_frame


fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Adjust codec as needed

out = cv2.VideoWriter("dehazed_video.mp4", fourcc, 20.0, (600, 400))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    dehazed_frame = defog_dcp(frame)  # Replace with your chosen algorithm

    out.write(dehazed_frame)
    cv2.imshow("Dehazed Video", dehazed_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
