from ultralytics import YOLO

# import cv2

# Load a model
model = YOLO("yolov8l-pose.pt")

# source = "prathamesh.jpg"

# results = model(source)

# plotted_results = results[0].plot()

# save a image with the plotted results
# cv2.imwrite("result.jpg", plotted_results)
results = model.predict(source=0, show=True, stream=True)
for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs

print(results)
