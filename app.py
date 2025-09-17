from flask import Flask, request, jsonify
import os, cv2, numpy as np, pymysql, json, requests
from tensorflow.keras.models import load_model
from datetime import datetime

app = Flask(__name__)

# --------------------------
# Base path ของโปรเจกต์
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------
# DB config (แก้ให้ตรงกับ DB ภายนอกของคุณ)
db_config = {
    'host': 'YOUR_DB_HOST',  # เช่น db123.host.com
    'user': 'cedubruc_attendance_system',
    'password': 'LS46s3Ue4w75YUdCr9Qd',
    'database': 'cedubruc_attendance_system',
    'charset': 'utf8mb4'
}

# --------------------------
# โหลด label_map.json จาก Google Drive หากยังไม่มี
LABEL_FILE = os.path.join(BASE_DIR, 'label_map.json')
LABEL_URL = "https://drive.google.com/uc?export=download&id=1xI1LYLUoAGdFmj7VXLI0fiZUuEIi_Fpy"
if not os.path.exists(LABEL_FILE):
    print("ดาวน์โหลด label_map.json...")
    r = requests.get(LABEL_URL)
    with open(LABEL_FILE, "wb") as f:
        f.write(r.content)

with open(LABEL_FILE, 'r', encoding='utf-8') as f:
    label_map = json.load(f)

# --------------------------
# โหลด model
MODEL_PATH = os.path.join(BASE_DIR, 'face_recognition_model.h5')
model = load_model(MODEL_PATH)

# --------------------------
# Face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --------------------------
# Map student name -> student_id
conn = pymysql.connect(**db_config, cursorclass=pymysql.cursors.DictCursor)
students_map = {}
with conn.cursor() as cursor:
    cursor.execute("SELECT student_id, name FROM students")
    for row in cursor.fetchall():
        students_map[row['name']] = row['student_id']
conn.close()

# --------------------------
# ฟังก์ชันบันทึก attendance
def save_attendance(student_id, course_id=11110):
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    conn = pymysql.connect(**db_config, cursorclass=pymysql.cursors.DictCursor)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM attendance WHERE student_id=%s AND course_id=%s AND date=%s",
                           (student_id, course_id, today_str))
            if cursor.fetchone():
                return "already"
            cursor.execute("INSERT INTO attendance (student_id, course_id, date, time_in, status) VALUES (%s,%s,%s,%s,'present')",
                           (student_id, course_id, today_str, time_str))
            conn.commit()
            return "inserted"
    finally:
        conn.close()

# --------------------------
# Flask route สำหรับรับรูปและเช็คชื่อ
@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400
    
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    results = []

    for (x, y, w, h) in faces:
        face_img = cv2.resize(gray[y:y+h, x:x+w], (100,100)).reshape(1,100,100,1)/255.0
        pred = model.predict(face_img)
        class_idx = np.argmax(pred)
        confidence = float(np.max(pred))
        name = label_map.get(str(class_idx), "Unknown")

        if confidence > 0.8 and name in students_map:
            student_id = students_map[name]
            status = save_attendance(student_id)
            results.append({
                "student": name,
                "student_id": student_id,
                "confidence": confidence,
                "status": status
            })
        else:
            results.append({
                "student": name,
                "confidence": confidence,
                "status": "unknown"
            })

    return jsonify(results)

# --------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
