import os
from ultralytics import YOLO
import cv2
from datetime import datetime

IMAGES_DIR = os.path.join('.', 'images')

image_path = os.path.join('D:/TIF/Semester 5/Project Peternakan Kambing/detection/test/kambing (198).jpg')  # Ganti dengan jalur gambar yang ingin Anda deteksi

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Load a model
model = YOLO('D:/TIF/Semester 5/Project Peternakan Kambing/detection/runs/detect/train5/weights/best.pt')  # load a custom model

threshold = 0.5

# Baca gambar
image = cv2.imread(image_path)
H, W, _ = image.shape

# Lakukan deteksi objek pada gambar
results = model(image)[0]

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        label = results.names[int(class_id)].upper()
        cv2.putText(image, label, (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Buat nama berkas dengan timestamp
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

# Simpan gambar hasil deteksi ke direktori yang ditentukan
output_dir = 'D:/TIF/Semester 5/Project Peternakan Kambing/detection/hasil'
output_image_path = os.path.join(output_dir, f'kambing_detection_{timestamp}.jpg')

# Pastikan direktori output ada atau buat jika belum ada
os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(output_image_path, image)

cv2.destroyAllWindows()
