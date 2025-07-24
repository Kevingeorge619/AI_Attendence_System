import os
import cv2
import face_recognition
import pickle
from datetime import datetime

# ========== CONFIGURATION ==========
DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pkl"
TOLERANCE = 0.45  # Lower = stricter match
FRAME_RESIZE_SCALE = 0.25  # Speeds up processing
# ===================================


def encode_faces():
    known_encodings = []
    known_names = []

    for name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(person_dir):
            continue

        for file in os.listdir(person_dir):
            file_path = os.path.join(person_dir, file)
            try:
                image = face_recognition.load_image_file(file_path)
                boxes = face_recognition.face_locations(image)
                encodings = face_recognition.face_encodings(image, boxes)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(name)
                    print(f"[+] Encoded {file_path}")
            except Exception as e:
                print(f"[!] Error encoding {file_path}: {e}")

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((known_encodings, known_names), f)
    print(f"[✔] Encodings saved to {ENCODINGS_FILE}")


def recognize_faces():
    if not os.path.exists(ENCODINGS_FILE):
        print("[!] Encodings file not found. Run face data encoding first.")
        return

    with open(ENCODINGS_FILE, "rb") as f:
        known_encodings, known_names = pickle.load(f)

    attendance = {}

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting camera... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=TOLERANCE)
            name = "Unknown"

            if True in matches:
                matched_idx = matches.index(True)
                name = known_names[matched_idx]

                if name not in attendance:
                    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    attendance[name] = time_now
                    print(f"[✔] {name} marked present at {time_now}")

            top, right, bottom, left = [int(v / FRAME_RESIZE_SCALE) for v in face_location]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n[Attendance Report]")
    for student, timestamp in attendance.items():
        print(f"{student} - {timestamp}")


def main():
    print("1. Encode face data")
    print("2. Start face recognition attendance")
    choice = input("Enter choice (1/2): ").strip()

    if choice == "1":
        encode_faces()
    elif choice == "2":
        recognize_faces()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
