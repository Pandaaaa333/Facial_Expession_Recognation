import tensorflow as tf
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, ELU
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import os

# Định nghĩa đường dẫn đến thư mục chứa tập huấn luyện và tập kiểm tra
train_dir = r'dataset_split/train'
val_dir = r'dataset_split/val'

# Chuẩn bị dữ liệu từ thư mục (ảnh xám 48x48)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

# Xây dựng mô hình CNN
model = Sequential()

# Block 1
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(ELU())
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Block 2
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(ELU())
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Block 3
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(ELU())
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully connected layers
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(ELU())
model.add(Dropout(0.5))  

# Output layer
model.add(Dense(7, activation='softmax'))

# Biên dịch mô hình
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Huấn luyện mô hình
model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator
)

# Lưu mô hình sau khi huấn luyện
model.save("emotion_cnn_model.h5")
print("Mô hình đã được lưu thành công!")