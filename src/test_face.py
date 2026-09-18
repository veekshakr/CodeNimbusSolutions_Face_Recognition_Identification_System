import cv2
from insightface.app import FaceAnalysis

# Load the pretrained model
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(640, 640))

# Read image
image = cv2.imread("data/faces/test.jpeg")

if image is None:
    print("ERROR: Image not found.")
    exit()

# Detect faces
faces = app.get(image)

print("Number of faces detected:", len(faces))

if len(faces) == 0:
    print("No face detected.")
    exit()

# Display information about detected faces
for i, face in enumerate(faces):
    print(f"\nFace {i + 1}")
    print("Bounding box:", face.bbox)
    print("Embedding shape:", face.embedding.shape)

print("\nFace embedding generated successfully!")