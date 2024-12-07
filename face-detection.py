import cv2

# Informasi target
target_info = {
    "name": "Difa Ananta",
    "age": 23,
    "gender": "Male"
}

# Buka kamera
cap = cv2.VideoCapture(0)

# Inisialisasi deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi ke grayscale untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Pilih target wajah pertama yang terdeteksi
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Ambil wajah pertama
        target_face = (x, y, x + w, y + h)

        # Gambar kotak hijau di sekitar wajah target
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Tambahkan crosshair di tengah wajah target
        center_x = (x + x + w) // 2
        center_y = (y + y + h) // 2
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 0), 2)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 0), 2)

        # Tampilkan informasi target di atas wajah
        label = f"Name: {target_info['name']}, Age: {target_info['age']}, Gender: {target_info['gender']}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow("Deteksi Wajah Otomatis", frame)

    # Keluar dengan menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
