import cv2
import os
import numpy as np

# Danh sách 7 cảm xúc
emotions = {
    1: "angry",
    2: "disgust",
    3: "fear",
    4: "happy",
    5: "sad",
    6: "surprise",
    7: "neutral"
}

# Thư mục chứa dataset
output_dir = r'F:\ComputerVisionProject\code_chinh\data\test_data'

# Tạo thư mục nếu chưa có
for emotion in emotions.values():
    os.makedirs(os.path.join(output_dir, emotion), exist_ok=True)

# Cấu hình webcam và detector
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

current_emotion = None
image_count = {emotion: len(os.listdir(os.path.join(output_dir, emotion))) for emotion in emotions.values()}

print("=== HƯỚNG DẪN ===")
print("Bấm phím 1–7 để chọn cảm xúc:")
for key, value in emotions.items():
    print(f"{key}: {value.capitalize()}")
print("Bấm SPACE để chụp ảnh, ESC để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Hiển thị khuôn mặt
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Hiển thị cảm xúc hiện tại
    if current_emotion:
        cv2.putText(frame, f"Emotion: {current_emotion.upper()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Capture Emotions", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key in [49, 50, 51, 52, 53, 54, 55]:  # Phím 1-7
        emotion_index = key - 48
        current_emotion = emotions[emotion_index]
        print(f"Đã chọn cảm xúc: {current_emotion}")
    elif key == 32 and current_emotion and len(faces) > 0:  # SPACE
        (x, y, w, h) = faces[0]

        # Cắt khuôn mặt
        face = gray[y:y + h, x:x + w]

        # ====== Tiền xử lý ======
        # Histogram Equalization
        face = cv2.equalizeHist(face)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face = clahe.apply(face)

        # Resize về 48x48
        face = cv2.resize(face, (48, 48))

        # ====== Lưu ảnh ======
        count = image_count[current_emotion]
        filename = os.path.join(output_dir, current_emotion, f"{count}.jpg")
        cv2.imwrite(filename, face)
        image_count[current_emotion] += 1
        print(f"✅ Đã lưu ảnh: {filename} (shape: 48x48)")

cap.release()
cv2.destroyAllWindows()
