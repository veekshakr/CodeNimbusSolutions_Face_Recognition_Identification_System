import cv2
import json
import os
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

os.makedirs("data/database", exist_ok=True)

DATABASE_FILE = "data/database/embeddings.json"

if os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, "r") as f:
        database = json.load(f)
else:
    database = {}

name = input("Enter person's name: ").strip()

image_path = input("Enter image path: ").strip()

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Image not found.")
    exit()

faces = app.get(image)

if len(faces) == 0:
    print("ERROR: No face detected.")
    exit()

if len(faces) > 1:
    print("ERROR: Multiple faces detected. Use an image with one person.")
    exit()

embedding = faces[0].embedding

embedding = embedding / np.linalg.norm(embedding)

database[name] = embedding.tolist()

with open(DATABASE_FILE, "w") as f:
    json.dump(database, f, indent=4)

print(f"\nFace enrolled successfully for: {name}")
print(f"Database saved at: {DATABASE_FILE}")