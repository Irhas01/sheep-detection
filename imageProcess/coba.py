import os
from ultralytics import YOLO
import cv2
import numpy as np
from rembg import remove
from PIL import Image

def estimate_weight(length_mm, breadth_mm):
    c = 0.064443
    d = 0.010059
    weight = c * breadth_mm + d * length_mm
    return weight

# Load YOLO model for the first detection
model_first = YOLO('D:/TIF/Semester 5/Project Peternakan Kambing/detection/runs/detect/train/weights/best.pt')

# Path to the input image
input_path = ('D:/TIF/Semester 5/Project Peternakan Kambing/detection/test/kambing (198).jpg')

# Load the input image
input_image = cv2.imread(input_path)

# Perform object detection on the input image with the first YOLO model
results_first = model_first(input_image)[0]

# Set a detection threshold for the first detection
threshold_first = 0.5
# Create a mask for the detected object in the first detection
mask_first = np.zeros(input_image.shape[:2], dtype=np.uint8)

# Iterate through the detected objects in the first detection
for result_first in results_first.boxes.data.tolist():
    x1_first, y1_first, x2_first, y2_first, score_first, class_id_first = result_first

    if score_first > threshold_first:
        cv2.rectangle(input_image, (int(x1_first), int(y1_first)), (int(x2_first), int(y2_first)), (0, 255, 0), 4)
        label_first = results_first.names[int(class_id_first)].upper()
        cv2.putText(input_image, label_first, (int(x1_first), int(y1_first - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Create a mask for the detected object in the first detection
        mask_first[int(y1_first):int(y2_first), int(x1_first):int(x2_first)] = 255

# Apply the mask to remove the background from the input image in the first detection
input_image_no_bg_first = cv2.bitwise_and(input_image, input_image, mask=mask_first)

input_image_no_bg_pil = Image.fromarray(cv2.cvtColor(input_image_no_bg_first, cv2.COLOR_BGR2RGB))
input_image_no_bg_removed = remove(np.array(input_image_no_bg_pil))

# Load YOLO model for the second detection
model_second = YOLO('D:/TIF/Semester 5/Project Peternakan Kambing/detection/runs/detect/train5/weights/best.pt')

# Perform object detection on the input image with the second YOLO model
results_second = model_second(input_image_no_bg_removed)[0]
# results_second = model_second(input_image)[0]

# Set a detection threshold for the second detection
threshold_second = 0.5

# Create a mask for the detected object in the second detection
mask_second = np.zeros(input_image.shape[:2], dtype=np.uint8)

# Iterate through the detected objects in the second detection
for result_second in results_second.boxes.data.tolist():
    x1_second, y1_second, x2_second, y2_second, score_second, class_id_second = result_second

    if score_second > threshold_second:
        cv2.rectangle(input_image, (int(x1_second), int(y1_second)), (int(x2_second), int(y2_second)), (0, 255, 0), 4)
        label_second = results_second.names[int(class_id_second)].upper()
        cv2.putText(input_image, label_second, (int(x1_second), int(y1_second - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Create a mask for the detected object in the second detection
        mask_second[int(y1_second):int(y2_second), int(x1_second):int(x2_second)] = 255

        # Estimate the weight based on the contour of the detected object in the second detection
        contour_second, _ = cv2.findContours(mask_second, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contour_second:
            current_contour_second = max(contour_second, key=cv2.contourArea)
            x_second, y_second, w_second, h_second = cv2.boundingRect(current_contour_second)
            length_mm_second = w_second
            breadth_mm_second = h_second
            estimated_weight_second = estimate_weight(length_mm_second, breadth_mm_second)
            cv2.putText(input_image, "Sheep (Second Detection)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(input_image, f"Weight (Second Detection): {estimated_weight_second:.2f} kg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Apply the mask to remove the background from the input image in the second detection
input_image_no_bg_second = cv2.bitwise_and(input_image, input_image, mask=mask_second)

# Remove background using rembg library
input_image_no_bg_pil = Image.fromarray(cv2.cvtColor(input_image_no_bg_second, cv2.COLOR_BGR2RGB))
input_image_no_bg_removed = remove(np.array(input_image_no_bg_pil))

# Convert the processed image to grayscale
gray_second = cv2.cvtColor(input_image_no_bg_removed, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
gray_blurred_second = cv2.GaussianBlur(gray_second, (15, 15), 0)

# Apply Bilateral Filter
gray_filtered_second = cv2.bilateralFilter(gray_second, 15, 75, 75)

# Apply Median Blur
gray_median_blurred_second = cv2.medianBlur(gray_second, 5)

# Apply additional image processing (e.g., edge detection and morphological operations)
edges = cv2.Canny(gray_median_blurred_second, 0, 0)
kernel = np.ones((10, 10), np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((15, 15), np.uint8)
cleaned = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
segmented_frame = np.uint8(cleaned)
contours, _ = cv2.findContours(segmented_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Display the original image
cv2.imshow("First Detection Results", input_image_no_bg_first)
cv2.imshow("Original Image", input_image)

# Display Grayscale image
#cv2.imshow("Grayscale", gray_first)
cv2.imshow("Grayscale", gray_second)

# Display Binary image
ret, binary_image = cv2.threshold(gray_median_blurred_second, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary Image", binary_image)

# Display the final image with object detection, background removal, and weight estimation
cv2.imshow("Object Detection with Background Removal and Weight Estimation", input_image_no_bg_removed)

# Wait for user input and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()