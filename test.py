import tensorflow as tf
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

# Định nghĩa đường dẫn đến tập test
test_dir = r'datasett/test1'

# Chuẩn bị dữ liệu test
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),  # Đảm bảo kích thước ảnh đúng
    color_mode='grayscale',  # Chuyển về ảnh xám
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

# Tải mô hình đã huấn luyện
model = load_model("emotion_cnn_model.h5")

# Đánh giá mô hình trên tập test
test_loss, test_acc = model.evaluate(test_generator)

# In kết quả ra màn hình
print(f"Độ chính xác trên tập kiểm tra: {test_acc*100:.2f}%")
