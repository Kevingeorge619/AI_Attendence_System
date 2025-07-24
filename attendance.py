import cv2
import face_recognition
import pickle
import mysql.connector
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import threading

# Load encodings
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)
known_encodings = data["encodings"]
known_names = data["names"]

# DB setup
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="YOUR_DATABASE_NAME"
)
cursor = conn.cursor()

# Email credentials
EMAIL_SENDER = "YOUR_EMAIL_ID"
EMAIL_PASSWORD = "YOUR_APP_PASSWORD"

def send_email_to_parent(student_name, timestamp):
    cursor.execute("SELECT parent_email FROM students WHERE name = %s", (student_name,))
    result = cursor.fetchone()
    if not result:
        print(f"No email found for {student_name}")
        return

    parent_email = result[0]
    subject = "Attendance Notification"
    body = f"Dear Parent,\n\n{student_name} was marked present on {timestamp}.\n\nAI Attendance System"

    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = parent_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, parent_email, msg.as_string())
        server.quit()
        print(f"Email sent to {parent_email}")
    except Exception as e:
        print(f"Email failed: {e}")

def send_email_thread(student_name, timestamp):
    send_email_to_parent(student_name, timestamp)

# Track attendance and email sent status separately
attendance = {}
email_sent = set()

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encs = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encs, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unidentified"

        if True in matches:
            match_idx = matches.index(True)
            name = known_names[match_idx]

            if name not in attendance:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance[name] = timestamp

                cursor.execute("INSERT INTO attendance (name, timestamp) VALUES (%s, %s)", (name, timestamp))
                conn.commit()

                # Send email asynchronously only once per person
                if name not in email_sent:
                    threading.Thread(target=send_email_thread, args=(name, timestamp), daemon=True).start()
                    email_sent.add(name)

                label_text = f"{name} Marked Present"
                print(f"✅ {name} marked present at {timestamp}")

            else:
                label_text = f"{name} Marked Present"
        else:
            label_text = "Unidentified"

        top, right, bottom, left = [v * 4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
