from ultralytics import YOLO
import shutil

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Define path to the image file
source = 'D:/TIF/Semester 5/Project Peternakan Kambing/detection/test/kambing (25).jpg'

# Run inference on the source
results = model(source)  # list of Results objects
output_dir = "D:/TIF/Semester 5/Project Peternakan Kambing/detection/hasil"
shutil.copy(source, output_dir)  # Menyalin gambar ke direktori output