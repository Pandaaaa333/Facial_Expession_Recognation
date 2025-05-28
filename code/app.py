import sys
import cv2
import numpy as np
import time
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from keras._tf_keras.keras.models import load_model

# Tải mô hình và nhãn cảm xúc
model = load_model("emotion_cnn_model_1.h5")
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Tạo thư mục lưu ảnh nếu chưa có
if not os.path.exists("captured_images"):
    os.makedirs("captured_images")

class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion Recognition - PyQt5")
        self.setGeometry(100, 100, 800, 600)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)

        self.btn_start = QPushButton("Bắt đầu")
        self.btn_stop = QPushButton("Dừng lại")
        self.btn_capture = QPushButton("Chụp ảnh")

        self.btn_start.clicked.connect(self.start_video)
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_capture.clicked.connect(self.capture_image)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_stop)
        layout.addWidget(self.btn_capture)
        self.setLayout(layout)

        self.cap = None
        self.running = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.frame_count = 0
        self.last_emotion_label = "..."
        self.last_update_time = time.time()
        self.current_frame = None

    def start_video(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.timer.start(30)

    def stop_video(self):
        if self.running:
            self.running = False
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.video_label.clear()

    def update_frame(self):
        if self.cap and self.running:
            ret, frame = self.cap.read()
            if not ret:
                return

            self.current_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            self.frame_count += 1
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                face = face / 255.0
                face = np.reshape(face, (1, 48, 48, 1))

                if self.frame_count % 5 == 0:
                    prediction = model.predict(face, verbose=0)
                    emotion_label = class_labels[np.argmax(prediction)]

                    if (emotion_label != self.last_emotion_label or 
                        time.time() - self.last_update_time > 2):
                        self.last_emotion_label = emotion_label
                        self.last_update_time = time.time()

                label_to_display = self.last_emotion_label

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label_to_display, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def capture_image(self):
        if self.current_frame is not None:
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                QMessageBox.warning(self, "Lỗi", "Không phát hiện khuôn mặt!")
                return

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (48, 48))
                face_input = face_resized / 255.0
                face_input = np.reshape(face_input, (1, 48, 48, 1))

                prediction = model.predict(face_input, verbose=0)
                emotion_label = class_labels[np.argmax(prediction)]

                # Tạo tên file
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"captured_images/{emotion_label}_{timestamp}.jpg"

                # Lưu ảnh (RGB để không bị đen trắng)
                face_rgb = self.current_frame[y:y+h, x:x+w]
                cv2.imwrite(filename, face_rgb)

                QMessageBox.information(self, "Thành công", f"Đã lưu ảnh với nhãn: {emotion_label}")
                break
        else:
            QMessageBox.warning(self, "Lỗi", "Không có frame hiện tại để chụp!")

    def closeEvent(self, event):
        self.stop_video()

# Chạy ứng dụng
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec_())
