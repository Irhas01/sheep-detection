import cv2

# Gantilah 'nama_gambar.jpg' dengan nama gambar yang ingin Anda tampilkan labelingnya
nama_gambar = '0000f2101250b009.jpg'

# Membaca gambar menggunakan OpenCV
gambar = cv2.imread(nama_gambar)

# Koordinat objek yang telah Anda labeling (misalnya, x1, y1, x2, y2)
x1, y1, x2, y2 = 49, 116, 339, 681  # Contoh koordinat

# Menggambar kotak pada gambar untuk menampilkan labeling
cv2.rectangle(gambar, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Warna hijau, lebar garis 2

# Menampilkan gambar dengan labeling
cv2.imshow('Gambar dengan Labeling', gambar)
cv2.waitKey(0)
cv2.destroyAllWindows()