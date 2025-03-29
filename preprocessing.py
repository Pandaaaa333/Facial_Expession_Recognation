import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

dataset_path = "datasett/train"  # Sửa đường dẫn để dùng dấu gạch chéo '/'
train_output = "datasett/train1"
test_output = "datasett/test1"

# Tạo thư mục đích nếu chưa tồn tại
for folder in [train_output, test_output]:
    if not os.path.exists(folder):
        os.makedirs(folder)

labels = [label for label in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, label))]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for label_name in labels:
    label_path = os.path.join(dataset_path, label_name)
    
    # Tạo thư mục cho từng cảm xúc trong train1 và test1
    train_label_path = os.path.join(train_output, label_name)
    test_label_path = os.path.join(test_output, label_name)
    os.makedirs(train_label_path, exist_ok=True)
    os.makedirs(test_label_path, exist_ok=True)
    
    face_images = []
    
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        
        # Đọc ảnh grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Lỗi khi đọc ảnh: {img_path}")
            continue
        
        # Face detection
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face_roi = image[y:y+h, x:x+w]
            
            # Cân bằng sáng bằng Histogram Equalization
            face_roi = cv2.equalizeHist(face_roi)
            
            # Cân bằng độ sáng và tương phản bằng CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            face_roi = clahe.apply(face_roi)
            
            # Resize về 48x48
            face_roi = cv2.resize(face_roi, (48, 48))
            
            face_images.append(face_roi)
    
    # Chia dữ liệu thành train (80%) và test (20%)
    if face_images:  # Kiểm tra nếu có ảnh khuôn mặt
        train_faces, test_faces = train_test_split(face_images, test_size=0.2, random_state=42)
        
        # Lưu ảnh vào các thư mục tương ứng
        for idx, face in enumerate(train_faces):
            cv2.imwrite(os.path.join(train_label_path, f"{idx}.jpg"), face)
        for idx, face in enumerate(test_faces):
            cv2.imwrite(os.path.join(test_label_path, f"{idx}.jpg"), face)
    else:
        print(f"Không tìm thấy khuôn mặt trong thư mục: {label_path}")

print("Chia dữ liệu hoàn tất!")
