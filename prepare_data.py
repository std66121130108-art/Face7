import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split

# --------------------------
# กำหนด path แบบอัตโนมัติ
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
IMAGE_SIZE = (100, 100)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --------------------------
# ตรวจสอบว่า dataset มีหรือไม่
# --------------------------
if not os.path.exists(DATASET_PATH):
    print(f"❌ ไม่พบโฟลเดอร์ dataset ที่ {DATASET_PATH}")
    exit(1)

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
            (x, y, w, h) = faces[0]  # ใช้ใบหน้าแรก
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, IMAGE_SIZE)
            images.append(face_resized)
            labels.append(label)

# --------------------------
# แบ่ง train/test
# --------------------------
X = np.array(images).reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1) / 255.0
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# บันทึก dataset และ label_map
# --------------------------
np.savez(os.path.join(BASE_DIR, 'face_data.npz'), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

label_map = {i: name for i, name in enumerate(class_names)}
with open(os.path.join(BASE_DIR, 'label_map.json'), 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False)

print(f"✅ Dataset เตรียมเรียบร้อย! รูปทั้งหมด: {len(images)}")
print(f"   - Training: {len(X_train)}")
print(f"   - Testing : {len(X_test)}")
print(f"📝 บันทึก label_map.json แล้ว ที่ {BASE_DIR}")
