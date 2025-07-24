import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import os
import mysql.connector
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load your trained model
model = tf.keras.models.load_model("face_recognition_model")

# Image input size (must match training size)
img_height, img_width = 180, 180

# Get class names from dataset folder
dataset_path = "dataset"
class_names = sorted(os.listdir(dataset_path))
print("Class names:", class_names)

# Set confidence threshold
confidence_threshold = 0.7

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="aiproject"
)
cursor = conn.cursor()

# Email credentials (replace with your real info)
EMAIL_SENDER = "kevingeorge422@gmail.com"
EMAIL_PASSWORD = "aoar lmwz jchd oymi"  # Use App Password if 2FA enabled

def send_email_to_parent(student_name, timestamp):
    # Fetch parent email from DB
    cursor.execute("SELECT parent_email FROM students WHERE name = %s", (student_name,))
    result = cursor.fetchone()
    if not result:
        print(f"No parent email found for {student_name}")
        return

    parent_email = result[0]
    subject = "Attendance Notification"
    body = f"Dear Parent,\n\n{student_name} has been marked present on {timestamp}.\n\nBest regards,\nAI Attendance System"

    # Compose email
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = parent_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, parent_email, msg.as_string())
        server.quit()
        print(f"Email sent to {parent_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Track attendance (to avoid duplicates)
attendance = {}

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        label = "No Face Detected"
    else:
        # Resize whole frame for prediction
        img = cv2.resize(frame, (img_width, img_height))
        img_array = np.expand_dims(img / 255.0, axis=0)

        # Predict
        predictions = model.predict(img_array)
        confidence = np.max(predictions[0])
        predicted_index = np.argmax(predictions[0])

        if confidence > confidence_threshold and predicted_index < len(class_names):
            predicted_name = class_names[predicted_index]
            label = f"{predicted_name} ({confidence:.2f})"

            if predicted_name not in attendance:
                time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance[predicted_name] = time_now
                print(f"{predicted_name} marked present at {time_now}")

                cursor.execute("INSERT INTO attendance (name, timestamp) VALUES (%s, %s)",
                               (predicted_name, time_now))
                conn.commit()

                # Send email to parent
                send_email_to_parent(predicted_name, time_now)
            else:
                label = f"{predicted_name} (Attendance Already Marked)"
        else:
            label = "Unidentified User"

    # Display label on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0) if "User" not in label else (0, 0, 255), 2)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
conn.close()

print("Session ended. Attendance logged.")
