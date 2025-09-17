from flask import Flask, render_template, request, redirect, url_for
import pymysql

app = Flask(__name__)

# --- Database config ---
db_config = {
    'host': 'localhost',
    'user': 'root',       # เปลี่ยนเป็น user ของคุณ
    'password': '',       # เปลี่ยนเป็น password ของคุณ
    'database': 'attendance_system',
    'charset': 'utf8mb4'
}

# --- หน้าเว็บฟอร์มลงทะเบียน ---
@app.route("/", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        student_id = request.form.get("student_id")
        name = request.form.get("name")
        course = request.form.get("course")

        try:
            conn = pymysql.connect(**db_config)
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO students (student_id, name, course)
                    VALUES (%s, %s, %s)
                """
                cursor.execute(sql, (student_id, name, course))
                conn.commit()
            conn.close()
            return redirect(url_for('success', name=name))
        except Exception as e:
            return f"เกิดข้อผิดพลาด: {e}"

    return render_template("register.html")

# --- หน้าแสดงผลสำเร็จ ---
@app.route("/success/<name>")
def success(name):
    return f"นักเรียน {name} ลงทะเบียนเรียบร้อยแล้ว!"

if __name__ == "__main__":
    app.run(debug=True)
