import cv2
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.models import load_model

# Tải mô hình đã huấn luyện
model = load_model("emotion_cnn_model_1.h5")

# Danh sách nhãn cảm xúc
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Khởi tạo bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mở camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Chuyển đổi ảnh sang grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Cắt và chuẩn hóa ảnh khuôn mặt
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0  # Chuẩn hóa
        face = np.reshape(face, (1, 48, 48, 1))
        
        # Dự đoán cảm xúc
        prediction = model.predict(face)
        emotion_label = class_labels[np.argmax(prediction)]
        
        # Vẽ hình chữ nhật quanh mặt và hiển thị nhãn cảm xúc
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Hiển thị kết quả
    cv2.imshow('Real-time Emotion Recognition', frame)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
