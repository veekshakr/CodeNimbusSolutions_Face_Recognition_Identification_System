import cv2
import json
import numpy as np
from insightface.app import FaceAnalysis

# Load face recognition model
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

DATABASE_FILE = "data/database/embeddings.json"

# Load enrolled faces
with open(DATABASE_FILE, "r") as f:
    database = json.load(f)

# Get test image
image_path = input("Enter image path: ").strip()

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Image not found.")
    exit()

# Detect faces
faces = app.get(image)

if len(faces) == 0:
    print("No face detected.")
    exit()

print("Number of faces detected:", len(faces))

# Compare each detected face
for i, face in enumerate(faces):

    embedding = face.embedding
    embedding = embedding / np.linalg.norm(embedding)

    best_name = "Unknown"
    best_similarity = -1

    # Compare with enrolled embeddings
    for name, stored_embedding in database.items():

        stored_embedding = np.array(stored_embedding)
        stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)

        similarity = np.dot(embedding, stored_embedding)

        if similarity > best_similarity:
            best_similarity = similarity
            best_name = name

    # Recognition threshold
    threshold = 0.45

    if best_similarity >= threshold:
        result = best_name
    else:
        result = "Unknown"

    print(f"\nFace {i + 1}")
    print("Identified as:", result)
    print("Similarity:", round(float(best_similarity), 4))