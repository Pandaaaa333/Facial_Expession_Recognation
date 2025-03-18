import cv2
import os
import numpy as np

dataset_path = "dataSet"
labels = os.listdir(dataset_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

X, y = [], []

for label_idx, label_name in enumerate(labels):
    label_path = os.path.join(dataset_path, label_name)
    
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        
        # Grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Lỗi khi đọc ảnh: {img_path}")
            continue
        
        # Face detection
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y_, w, h) in faces:
            face_roi = image[y_:y_+h, x:x+w]
            
            # Cân bằng độ sáng và tương phản bằng phương pháp CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            face_roi = clahe.apply(face_roi)
            
            # Resize về 48x48
            face_roi = cv2.resize(face_roi, (48, 48))
            
            # Chuẩn hóa dữ liệu
            face_roi = np.expand_dims(face_roi, axis=-1)
            face_roi = face_roi / 255.0 
            
            # Lưu vào tập dữ liệu
            X.append(face_roi)
            y.append(label_idx)  


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("Dữ liệu đã tải xong!")
print(f"Số mẫu: {X.shape[0]}, Dữ liệu mẫu: {X.shape[1:]}")

