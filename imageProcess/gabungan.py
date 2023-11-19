import os
from ultralytics import YOLO
import cv2
from datetime import datetime
import numpy as np
# from rembg import remove
from PIL import Image

def estimate_weight(length_mm, breadth_mm):
    c = 0.064443
    d = 0.010059
    weight = c * breadth_mm + d * length_mm
    return weight

# Load YOLO model
model = YOLO('D:/TIF/Semester 5/Project Peternakan Kambing/detection/runs/detect/train5/weights/best.pt')

# Path to the input image
input_path = ('D:/TIF/Semester 5/Project Peternakan Kambing/detection/test/kambing (127).jpg')

# Load the input image
input_image = cv2.imread(input_path)

# Perform object detection on the input image with YOLO
results = model(input_image)[0]

# Set a detection threshold
threshold = 0.5

# Create a mask for the detected object
mask = np.zeros(input_image.shape[:2], dtype=np.uint8)

# Iterate through the detected objects
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(input_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        label = results.names[int(class_id)].upper()
        cv2.putText(input_image, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Create a mask for the detected object
        mask[int(y1):int(y2), int(x1):int(x2)] = 255

#   the mask to remove the background from the input image
input_image_no_bg = cv2.bitwise_and(input_image, input_image, mask=mask)

# Get the estimated weight of the detected object
gray = cv2.cvtColor(input_image_no_bg, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 0, 0)
kernel1 = np.ones((10, 10), np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel1)
kernel2 = np.ones((15, 15), np.uint8)
cleaned = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
segmented_frame = np.uint8(cleaned)
contours, _ = cv2.findContours(segmented_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    length_mm = w
    breadth_mm = h
    estimated_weight = estimate_weight(length_mm, breadth_mm)
    cv2.putText(input_image_no_bg, "Sheep", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(input_image_no_bg, f"Weight: {estimated_weight:.2f} kg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the final image with object detection, background removal, and weight estimation
cv2.imshow("Object Detection with Background Removal and Weight Estimation", input_image_no_bg)

# Wait for user input and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()