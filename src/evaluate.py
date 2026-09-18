import cv2
import json
import numpy as np
from insightface.app import FaceAnalysis

# Load model
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

DATABASE_FILE = "data/database/embeddings.json"

with open(DATABASE_FILE, "r") as f:
    database = json.load(f)


def identify(image_path, threshold=0.45):
    image = cv2.imread(image_path)

    if image is None:
        return "Image not found", 0.0

    faces = app.get(image)

    if len(faces) == 0:
        return "No face detected", 0.0

    embedding = faces[0].embedding
    embedding = embedding / np.linalg.norm(embedding)

    best_name = "Unknown"
    best_similarity = -1

    for name, stored_embedding in database.items():
        stored_embedding = np.array(stored_embedding)
        stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)

        similarity = np.dot(embedding, stored_embedding)

        if similarity > best_similarity:
            best_similarity = similarity
            best_name = name

    if best_similarity >= threshold:
        return best_name, float(best_similarity)

    return "Unknown", float(best_similarity)


# Test cases
test_cases = [
    ("Known Face", "data/faces/test.jpeg", "Veeksha"),
    ("Unknown Face", "data/faces/unknown.jpeg", "Unknown")
]

correct = 0

print("\n===== FACE RECOGNITION EVALUATION =====\n")

for test_name, image_path, expected in test_cases:

    predicted, similarity = identify(image_path)

    status = "PASS" if predicted == expected else "FAIL"

    if status == "PASS":
        correct += 1

    print(f"Test: {test_name}")
    print(f"Expected: {expected}")
    print(f"Predicted: {predicted}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Result: {status}")
    print("-" * 40)

accuracy = (correct / len(test_cases)) * 100

print(f"\nEvaluation Accuracy: {accuracy:.2f}%")