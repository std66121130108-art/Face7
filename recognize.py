import sys, os, cv2, numpy as np, pymysql, json, requests, gdown
from tensorflow.keras.models import load_model
from datetime import datetime
from time import time

# --------------------------
# Base path ของ project
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------
# รับ course_id จาก args
# --------------------------
COURSE_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 11110
print(f"เริ่มเช็คชื่อวิชา: {COURSE_ID}")

# --------------------------
# โหลด label_map.json จาก GitHub
# --------------------------
LABEL_URL = 'https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/label_map.json'  # แก้ลิงค์ตาม repo จริง
try:
    r = requests.get(LABEL_URL)
    r.raise_for_status()
    label_map = r.json()
    print("โหลด label_map.json จาก GitHub เรียบร้อย")
except Exception as e:
    print("Error โหลด label_map.json:", e)
    label_map = {}

# --------------------------
# โหลดโมเดลจาก Google Drive ถ้าไฟล์ยังไม่มี
# --------------------------
FACE_MODEL_PATH = os.path.join(BASE_DIR, 'face_model.h5')
RECOG_MODEL_PATH = os.path.join(BASE_DIR, 'face_recognition_model.h5')

# ใส่ Google Drive ID ของไฟล์
FACE_MODEL_ID = '1isj1GNME9E_8glCfM0UCeaCLtUVqhd3V'
RECOG_MODEL_ID = 'YOUR_RECOG_MODEL_FILE_ID'  # ใส่ไฟล์อีกตัว

if not os.path.exists(FACE_MODEL_PATH):
    print("ดาวน์โหลด face_model.h5 จาก Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id={FACE_MODEL_ID}', FACE_MODEL_PATH, quiet=False)

if not os.path.exists(RECOG_MODEL_PATH):
    print("ดาวน์โหลด face_recognition_model.h5 จาก Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id={RECOG_MODEL_ID}', RECOG_MODEL_PATH, quiet=False)

# --------------------------
# โหลดโมเดล
# --------------------------
model = load_model(RECOG_MODEL_PATH)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --------------------------
# DB config จาก Environment Variables
# --------------------------
db_config = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', ''),
    'database': os.environ.get('DB_NAME', 'attendance_system'),
    'charset': 'utf8mb4'
}

# --------------------------
# ฟังก์ชันบันทึก attendance
# --------------------------
def save_attendance(student_id):
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    conn = pymysql.connect(**db_config, cursorclass=pymysql.cursors.DictCursor)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM attendance WHERE student_id=%s AND course_id=%s AND date=%s",
                           (student_id, COURSE_ID, today_str))
            if cursor.fetchone():
                return "already"
            cursor.execute("INSERT INTO attendance (student_id, course_id, date, time_in, status) VALUES (%s,%s,%s,%s,'present')",
                           (student_id, COURSE_ID, today_str, time_str))
            conn.commit()
            return "inserted"
    finally:
        conn.close()

# --------------------------
# ดึง student_map จาก DB
# --------------------------
conn = pymysql.connect(**db_config, cursorclass=pymysql.cursors.DictCursor)
students_map = {}
with conn.cursor() as cursor:
    cursor.execute("SELECT student_id, name FROM students")
    for row in cursor.fetchall():
        students_map[row['name']] = row['student_id']
conn.close()

# --------------------------
# เปิด Video file หรือ webcam (deploy ต้องใช้ video file)
# --------------------------
VIDEO_PATH = os.environ.get('VIDEO_PATH')  # กำหนด path ของ video สำหรับ Render
if VIDEO_PATH:
    cap = cv2.VideoCapture(VIDEO_PATH)
else:
    cap = cv2.VideoCapture(0)  # สำหรับทดสอบ local

COOLDOWN = 60  # วินาที กัน spam insert DB
last_seen = {}

# --------------------------
# ตัวแปรข้อความแจ้งเตือน
# --------------------------
message = ""
message_time = 0
MESSAGE_DURATION = 2

# --------------------------
# ตัวแปร countdown ปิดกล้อง
# --------------------------
countdown_start = None
countdown_seconds = 5
last_person = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    current_person = None

    for (x, y, w, h) in faces:
        face_img = cv2.resize(gray[y:y+h, x:x+w], (100,100)).reshape(1,100,100,1)/255.0
        pred = model.predict(face_img)
        class_idx = np.argmax(pred)
        confidence = np.max(pred)
        name = label_map.get(str(class_idx), "Unknown")
        current_person = name

        if confidence > 0.80 and name in students_map:
            student_id = students_map[name]
            now_time = time()

            result = save_attendance(student_id)

            if student_id not in last_seen or now_time - last_seen[student_id] > COOLDOWN:
                last_seen[student_id] = now_time

            if result == "inserted":
                message = f"{name} Present ✅"
                message_time = time()
                print(f"{name} -> inserted ({confidence*100:.2f}%)")
                countdown_start = None
                last_person = name
            elif result == "already":
                message = f"{name} Already Checked In ❌"
                message_time = time()
                print(f"{name} -> already ({confidence*100:.2f}%)")
                if last_person != name:
                    countdown_start = time()
                    last_person = name
                elif countdown_start is None:
                    countdown_start = time()

        color = (0,255,0) if confidence>0.80 else (0,0,255)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame, f"{name} {confidence*100:.2f}%", (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    # --------------------------
    # แสดงข้อความแจ้งเตือน
    # --------------------------
    if message and time() - message_time < MESSAGE_DURATION:
        (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        x = (frame.shape[1] - tw) // 2
        y = 80
        cv2.putText(frame, message, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
    else:
        message = ""

    # --------------------------
    # แสดง countdown ปิดกล้อง
    # --------------------------
    if countdown_start:
        elapsed = time() - countdown_start
        remaining = countdown_seconds - int(elapsed)
        if remaining > 0:
            (tw, th), _ = cv2.getTextSize(f"Closing in {remaining}", cv2.FONT_HERSHEY_SIMPLEX, 2, 5)
            x = (frame.shape[1] - tw) // 2
            y = frame.shape[0] // 2
            cv2.putText(frame, f"Closing in {remaining}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
        else:
            print("ครบเวลา -> ปิดกล้องแล้ว...")
            break

    cv2.imshow("Face Recognition Attendance", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("กดออก -> ปิดกล้องแล้ว...")
        break

cap.release()
cv2.destroyAllWindows()
print("Q")
