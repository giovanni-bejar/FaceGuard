import cv2
import os
import time
import torch
import numpy as np
import pickle
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import shutil
from collections import deque
import ctypes

# Initialize the MTCNN and InceptionResnetV1 model
mtcnn = MTCNN()
model = InceptionResnetV1(pretrained='vggface2').eval()

# File paths for saving and loading known embeddings and names
embeddings_file = 'known_embeddings.npy'
names_file = 'known_names.pkl'

# Load or initialize known embeddings and names
if os.path.exists(embeddings_file) and os.path.exists(names_file):
    with open(names_file, 'rb') as f:
        known_names = pickle.load(f)
    known_embeddings = np.load(embeddings_file, allow_pickle=True).item()
else:
    known_embeddings = {}
    known_names = []

def save_face_crop(name, face_crop):
    """Save the cropped face in the specified directory."""
    directory = f"Database/{name}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    i = 1
    while os.path.exists(f"{directory}/{i}.jpg"):
        i += 1
    face_crop.save(f"{directory}/{i}.jpg")

def delete_face():
    """Delete the face data of a specified person."""
    global known_names, known_embeddings
    if not known_names:
        print("No faces available to delete.")
        return
    unique_names = set(known_names)
    for idx, name in enumerate(unique_names, start=1):
        print(f"{idx}. {name}")
    choice = input("Enter the number of the face you want to delete (or 'q' to cancel): ")
    if choice.lower() == 'q':
        return
    try:
        choice = int(choice)
        if choice < 1 or choice > len(unique_names):
            raise ValueError()
    except ValueError:
        print("Invalid choice. Please enter a valid number or 'q' to cancel.")
        return
    name_to_delete = list(unique_names)[choice - 1]
    shutil.rmtree(os.path.join("Database", name_to_delete), ignore_errors=True)
    known_names = [name for name in known_names if name != name_to_delete]
    known_embeddings.pop(name_to_delete, None)
    with open(names_file, 'wb') as f:
        pickle.dump(known_names, f)
    np.save(embeddings_file, known_embeddings)
    print(f"Face data for {name_to_delete} has been deleted.")

def get_embedding(face):
    """Get the embedding of the face using the InceptionResnetV1 model."""
    with torch.no_grad():
        return model(face.unsqueeze(0))

def train_new_face():
    """Train the model with a new face."""
    name = input("Enter the name of the person: ")
    os.makedirs(os.path.join("Database", name), exist_ok=True)
    cap = cv2.VideoCapture(0)
    print("Collecting face samples. Press 'q' to stop.")
    last_capture_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face_crop_pil = Image.fromarray(frame[y1:y2, x1:x2]).resize((160, 160))
                if time.time() - last_capture_time > 1:
                    last_capture_time = time.time()
                    face_crop_pil.save(os.path.join("Database", name, f"{int(last_capture_time)}.png"))
                    embedding = get_embedding(torch.tensor(np.array(face_crop_pil)).permute(2, 0, 1).float() / 255.0)
                    known_names.append(name)
                    known_embeddings[name] = embedding.numpy()
                    with open(names_file, 'wb') as f:
                        pickle.dump(known_names, f)
                    np.save(embeddings_file, known_embeddings)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def lock_screen():
    """Lock the screen."""
    ctypes.windll.user32.LockWorkStation()

def playground():
    """Main function to detect and recognize faces."""
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    distance_threshold = 0.9
    detection_history = deque(maxlen=5)
    unrecognized_face_timer = 0

    def most_frequent(lst):
        """Returns the most frequent element in the list."""
        return max(set(lst), key=lst.count)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, _ = mtcnn.detect(frame)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                # Ensure the coordinates are within the bounds of the frame
                h, w, _ = frame.shape
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
                face_crop_pil = Image.fromarray(frame[y1:y2, x1:x2]).resize((160, 160))
                face_tensor = torch.tensor(np.array(face_crop_pil)).permute(2, 0, 1).float() / 255.0
                embedding = get_embedding(face_tensor)
                distances = []
                for known_embedding in known_embeddings.values():
                    distance = np.linalg.norm(embedding - known_embedding)
                    distances.append(distance)
                if distances:
                    min_distance_idx = np.argmin(distances)
                    recognized_name = known_names[min_distance_idx] if distances[min_distance_idx] < distance_threshold else "Unknown"
                    detection_history.append(recognized_name)
                    final_recognition = most_frequent(detection_history)
                    if final_recognition == "Unknown":
                        unrecognized_face_timer += 1
                    else:
                        unrecognized_face_timer = 0
                        # Save the recognized face to the database
                        current_time = int(time.time())
                        face_crop_pil.save(os.path.join("Database", final_recognition, f"{current_time}.png"))
                else:
                    unrecognized_face_timer += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, final_recognition, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            unrecognized_face_timer += 1

        if unrecognized_face_timer >= 5 * fps:
            lock_screen()
            break

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main menu of the program."""
    while True:
        print("1. Train New Face")
        print("2. Delete Face")
        print("3. Start Face Recognition")
        print("4. Exit")
        choice = input("Enter your choice: ")
        if choice == '1':
            train_new_face()
        elif choice == '2':
            delete_face()
        elif choice == '3':
            playground()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()
