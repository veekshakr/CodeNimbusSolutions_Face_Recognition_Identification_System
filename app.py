import streamlit as st
import cv2
import json
import numpy as np
from insightface.app import FaceAnalysis

st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="centered"
)

st.title("👤 Face Recognition Identification System")
st.write("Upload a face image to identify an enrolled person.")

# Load model
@st.cache_resource
def load_model():
    model = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model

app = load_model()

# Load database
@st.cache_data
def load_database():
    with open("data/database/embeddings.json", "r") as f:
        return json.load(f)

database = load_database()

# Similarity threshold
threshold = st.slider(
    "Recognition Threshold",
    min_value=0.30,
    max_value=0.70,
    value=0.45,
    step=0.01
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Identify Face"):

        faces = app.get(image)

        if len(faces) == 0:
            st.error("No face detected.")
        
        else:

            st.success(f"{len(faces)} face(s) detected.")

            for i, face in enumerate(faces):

                embedding = face.embedding
                embedding = embedding / np.linalg.norm(embedding)

                best_name = "Unknown"
                best_similarity = -1

                for name, stored_embedding in database.items():

                    stored_embedding = np.array(stored_embedding)
                    stored_embedding = (
                        stored_embedding /
                        np.linalg.norm(stored_embedding)
                    )

                    similarity = np.dot(
                        embedding,
                        stored_embedding
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_name = name

                st.subheader(f"Face {i + 1}")

                if best_similarity >= threshold:
                    st.success(
                        f"Identified as: **{best_name}**"
                    )
                else:
                    st.warning(
                        "Identified as: **Unknown**"
                    )

                st.write(
                    f"Similarity Score: "
                    f"**{best_similarity:.4f}**"
                )

                st.write(
                    f"Threshold: **{threshold:.2f}**"
                )