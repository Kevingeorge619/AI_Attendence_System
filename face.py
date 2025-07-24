import cv2
import os
import face_recognition

# Ask for person name
name = input("Enter the name of the person: ").strip()

# Create directory
save_path = os.path.join("dataset", name)
os.makedirs(save_path, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
count = 0
max_images = 20

print("📷 Capturing images with face detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to access camera.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)

    # Draw rectangle around face and save image if found
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Crop face
        face_img = frame[top:bottom, left:right]
        img_path = os.path.join(save_path, f"{name}_{count}.jpg")
        cv2.imwrite(img_path, face_img)
        print(f"✅ Saved: {img_path}")
        count += 1

    # Show preview
    cv2.imshow("Capturing Face (Press Q to Quit)", frame)

    # Break if enough images
    if count >= max_images or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Face data collection done.")
