import pymysql
from datetime import datetime

try:
    connection = pymysql.connect(
        host='localhost',
        user='root',      # เปลี่ยนให้ตรงกับ MySQL ของสหาย
        password='',      # ใส่รหัสผ่าน MySQL
        database='attendance_system',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

    with connection.cursor() as cursor:
        sql = "INSERT INTO attendance (student_id, name, course, timestamp) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, ("650001", "สมชาย ใจดี", "วิชา AI เบื้องต้น", datetime.now()))
    connection.commit()

    print("บันทึกข้อมูลสำเร็จ!")

except Exception as e:
    print("เกิดข้อผิดพลาด:", e)

finally:
    connection.close()
