import face_recognition
import os
import pickle

# Folder structure: dataset/Name/image.jpg
dataset_path = "dataset"
encodings = []
names = []

for person in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person)
    if not os.path.isdir(person_folder):
        continue
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        image = face_recognition.load_image_file(img_path)
        face_encs = face_recognition.face_encodings(image)
        if face_encs:
            encodings.append(face_encs[0])
            names.append(person)
        else:
            print(f"No face found in {img_path}")

# Save encodings to file
with open("encodings.pkl", "wb") as f:
    pickle.dump({"encodings": encodings, "names": names}, f)

print("✅ Encodings saved to encodings.pkl")
