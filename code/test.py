import tensorflow as tf
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Định nghĩa đường dẫn đến tập test
test_dir = r'F:\ComputerVisionProject\code_chinh\data\test_data'

# Chuẩn bị dữ liệu test
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),  # Đảm bảo kích thước ảnh đúng
    color_mode='grayscale',  # Chuyển về ảnh xám
    batch_size=64,
    class_mode='categorical',
    shuffle=False  # Không xáo trộn để giữ đúng thứ tự nhãn
)

# Tải mô hình đã huấn luyện
model = load_model(r'F:\ComputerVisionProject\code_chinh\trained_model\emotion_cnn_model_1.h5')

# Dự đoán nhãn cho tập test
y_pred_prob = model.predict(test_generator)  # Dự đoán xác suất
y_pred = np.argmax(y_pred_prob, axis=1)  # Lấy nhãn có xác suất cao nhất

# Lấy nhãn thực tế từ test_generator
y_true = test_generator.classes

# Tính Accuracy, Precision, Recall, F1-score
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# In các tiêu chí đánh giá
print("\nĐánh giá mô hình:")
print(f" Accuracy: {accuracy:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall: {recall:.4f}")
print(f" F1-score: {f1:.4f}")



