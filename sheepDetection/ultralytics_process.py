import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# Fungsi untuk menghitung perkiraan berat
def estimate_weight(length_mm, breadth_mm):
    c = 0.064443
    d = 0.010059
    weight = c * breadth_mm + d * length_mm
    return weight

# Memuat model YOLO
model = YOLO('D:/Backup agim/document/All Project/Kuliah/semester 5/Python/sheepDetection/runs/detect/train5/weights/best.pt')

# Fungsi untuk memproses gambar
def process_image(image_bytes, id):
    try:
        # Mengonversi byte gambar ke format OpenCV
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        input_image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Melakukan deteksi objek dengan YOLO
        results = model(input_image)[0]

        # Menetapkan ambang deteksi
        threshold = 0.5

        # Membuat masker untuk objek yang terdeteksi
        mask = np.zeros_like(input_image, dtype=np.uint8)
        # Loop melalui objek yang terdeteksi
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                # Membuat persegi panjang untuk objek yang terdeteksi
                cv2.rectangle(input_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                label = results.names[int(class_id)].upper()
                cv2.putText(mask, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

                mask[int(y1):int(y2), int(x1):int(x2)] = 255    

        # Menggunakan masker untuk menghapus latar belakang dari gambar input
        _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        binary_mask = cv2.convertScaleAbs(binary_mask)  # Convert to CV_8U if needed
        binary_mask = cv2.resize(binary_mask, (input_image.shape[1], input_image.shape[0]))  # Resize if needed
        input_image_no_bg = cv2.bitwise_and(input_image, input_image, mask=binary_mask)

        # Mendapatkan perkiraan berat objek yang terdeteksi
        gray = cv2.cvtColor(input_image_no_bg, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 0, 0)
        kernel = np.ones((10, 10), np.uint8)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((15, 15), np.uint8)
        cleaned = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
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

            # Menampilkan berat perkiraan
            print("Berat Perkiraan:", estimated_weight, "kg")

            # Menyimpan hasil masker objek terdeteksi
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_path = f'uploads/images/{id}_{timestamp}.jpeg'
            cv2.imwrite(file_path, input_image_no_bg)

            return file_path  # Mengembalikan path file yang disimpan

    except Exception as e:
        print(f"Error: {str(e)}")

# Contoh penggunaan dengan gambar dari API
# Anda perlu menggantinya dengan mekanisme yang sesuai dengan API Anda
# Misalnya, jika menggunakan Flask, Anda dapat menggunakan request untuk mengambil gambar dari endpoint API
# response = requests.get('url_api_gambar')
# image_bytes_from_api = response.content

# Contoh penggunaan fungsi process_image dengan gambar dari API
# process_image(image_bytes_from_api)
