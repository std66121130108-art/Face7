import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

# ✅ Path ไปยัง dataset
dataset_path = 'C:/xampp/htdocs/projectFace/FaceRecognitionProject/dataset'

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"ไม่พบโฟลเดอร์ dataset ที่: {dataset_path}")

# 📦 เตรียมข้อมูลภาพ
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 🧠 โหลด MobileNetV2 แบบไม่รวมหัวโมเดล (include_top=False)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 🔒 Freeze layers ของ base model
for layer in base_model.layers:
    layer.trainable = False

# 🧱 ต่อหัวโมเดลใหม่
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ⚙️ คอมไพล์โมเดล
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 💾 เซฟโมเดลที่ดีที่สุด
checkpoint = ModelCheckpoint(
    'face_model_transfer.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# 🚀 เทรนโมเดล
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

print("✅ เทรนเสร็จแล้วด้วย Transfer Learning และเซฟไว้ใน face_model_transfer.h5")
