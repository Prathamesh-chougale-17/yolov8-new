from ultralytics import YOLO

# from ultralytics.yolo.v8.detect.predict import Detection
# Load a model
model = YOLO("yolov8n.pt")  # build a new model from scratch
# import cv2

# Use the model
results = model.train(data="config.yaml", epochs=1)

# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# results = model.predict(source="0", show=True, stream=True)
# for r in results:
#     boxes = r.boxes  # Boxes object for bbox outputs
#     masks = r.masks  # Masks object for segment masks outputs
#     probs = r.probs  # Class probabilities for classification outputs

print(results)
