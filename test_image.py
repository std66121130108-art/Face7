import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --------------------------
# Path dataset
# --------------------------
DATASET_PATH = "C:/xampp/htdocs/projectFace/FaceRecognitionProject/dataset"
IMAGE_SIZE = (100, 100)

# --------------------------
# โหลด dataset
# --------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
images, labels = [], []
class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])

for label, person_name in enumerate(class_names):
    person_path = os.path.join(DATASET_PATH, person_name)
    for img_name in os.listdir(person_path):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(50, 50))
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, IMAGE_SIZE)
            images.append(face_resized)
            labels.append(label)

X = np.array(images).reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1) / 255.0
y = to_categorical(np.array(labels), num_classes=len(class_names))

# --------------------------
# แบ่ง train/test
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# สร้างโมเดล CNN
# --------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --------------------------
# Train model
# --------------------------
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=8)

# --------------------------
# บันทึกโมเดล + label map
# --------------------------
model.save('C:/xampp/htdocs/projectFace/FaceRecognitionProject/face_recognition_model.h5')

label_map = {i: name for i, name in enumerate(class_names)}
import json
with open('C:/xampp/htdocs/projectFace/FaceRecognitionProject/label_map.json', 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False)

print("✅ Train เสร็จแล้ว! โมเดลและ label_map ถูกบันทึกเรียบร้อย")
