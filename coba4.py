from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('D:/TIF/Semester 5/Project Peternakan Kambing/detection/runs/detect/train4/weights/best.pt')  # load a custom model

# Predict with the model
results = model('https://cff2.earth.com/uploads/2023/02/09061922/Sheep-960x640.jpg')