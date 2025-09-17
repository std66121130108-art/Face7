import os
import numpy as np
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage
from tensorflow.keras.models import load_model
import cv2
from io import BytesIO
import pymysql
from datetime import datetime
import pymysql

try:
    connection = pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="attendance_system",
        cursorclass=pymysql.cursors.DictCursor
    )
    print("เชื่อมต่อฐานข้อมูลสำเร็จ")
except Exception as e:
    print("เชื่อมต่อฐานข้อมูลไม่สำเร็จ:", e)


app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = 'qnU3pYuGFahNRMWTwvlib3yIWq4VcGCUazqCZRruK5Ul5ExoErS+ToIoOTzcWgh27RNwtDaD+jkEwxQQ4ihBMsxxIJ44r8awy27jxfxHLzfSE8Q5Ol5B9RQU4NkJbYgF4Rb2nuTugnjxojHzMPsi/gdB04t89/1O/w1cDnyilFU='
LINE_CHANNEL_SECRET = 'fe84396096ee0af9d03a354395623727'

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# โหลดโมเดลและ cascade
model = load_model('face_recognition_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

DATASET_PATH = 'dataset'
folder_names = sorted([name for name in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, name))])
label_map = {i: folder_names[i] for i in range(len(folder_names))}

# *** Database connection config ***
db_config = {
    'host': 'localhost',
    'user': 'root',  # เปลี่ยนเป็น user ของคุณ
    'password': '',  # เปลี่ยนเป็น password ของคุณ
    'database': 'attendance_db',
    'charset': 'utf8mb4'
}

# ตัวอย่าง mapping name -> student info (แก้เป็นฐานข้อมูลจริงหรือดึงจาก DB ก็ได้)
student_info_map = {
    'นายสมชาย': {'student_id': 'S001', 'course': 'วิทยาการคอมพิวเตอร์'},
    'นางสาวสุนีย์': {'student_id': 'S002', 'course': 'วิศวกรรมคอมพิวเตอร์'},
    # เพิ่มคนอื่นๆ ตาม dataset
}

def save_attendance_to_db(student_id, name, course):
    try:
        conn = pymysql.connect(**db_config)
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO attendance (student_id, name, course, timestamp)
                VALUES (%s, %s, %s, NOW())
            """
            cursor.execute(sql, (student_id, name, course))
            conn.commit()
        conn.close()
        print(f"บันทึกข้อมูลสำเร็จ: {student_id}, {name}, {course}")
    except Exception as e:
        print("Error saving to DB:", e)

def recognize_face_from_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return "ไม่พบใบหน้าในภาพ"

    (x, y, w, h) = faces[0]
    face_img = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_img, (100, 100))
    face_input = face_resized.reshape(1, 100, 100, 1) / 255.0

    prediction = model.predict(face_input)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    if confidence < 0.5:
        return "ไม่รู้จักใบหน้านี้"
    else:
        name = label_map.get(predicted_class, "Unknown")
        # ถ้ารู้จัก ให้บันทึกเข้า DB
        info = student_info_map.get(name)
        if info:
            save_attendance_to_db(info['student_id'], name, info['course'])
            return f"ใบหน้านี้คือ {name} (ความมั่นใจ {confidence*100:.2f}%)\nข้อมูลการเข้าเรียนถูกบันทึกแล้ว"
        else:
            return f"ใบหน้านี้คือ {name} (ความมั่นใจ {confidence*100:.2f}%)\nแต่ไม่พบข้อมูลนักเรียนในระบบ"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_id = event.message.id
    message_content = line_bot_api.get_message_content(message_id)
    image_binary = BytesIO(message_content.content)

    img_array = np.frombuffer(image_binary.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    result_text = recognize_face_from_image(img)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=result_text)
    )

@handler.add(MessageEvent)
def handle_message(event):
    # ถ้าไม่ใช่รูปภาพ ตอบบอกรองรับแค่รูปภาพ
    if not isinstance(event.message, ImageMessage):
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="ตอนนี้รองรับเฉพาะรูปภาพเท่านั้นครับ")
        )

if __name__ == "__main__":
    app.run(port=5000, debug=True)
