# Face Recognition Identification System

A face recognition system developed as part of the Code Nimbus Solutions AI/ML internship assignment.

The system can enroll a person's face, generate a face embedding, store it locally, and identify a new face by comparing it with enrolled faces. If the similarity score is below the selected threshold, the system returns **Unknown**.

## What the Project Does

The project has two main functions:

* Face enrollment
* Face identification

During enrollment, the system detects a face in an image, generates its face embedding, and stores the embedding with the person's name.

During identification, a new face is detected and its embedding is compared with the stored embeddings using cosine similarity.

## Technologies Used

* Python
* InsightFace
* ONNX Runtime
* OpenCV
* NumPy
* Scikit-learn
* Streamlit

## How It Works

The basic workflow is:

```text
Input Image
     |
     v
Face Detection
     |
     v
Face Embedding Generation
     |
     v
Compare with Stored Embeddings
     |
     v
Cosine Similarity
     |
     +----------------------+
     |                      |
     v                      v
Similarity >= Threshold    Similarity < Threshold
     |                      |
     v                      v
Person Name               Unknown
```

## Face Detection and Embeddings

The project uses the `buffalo_l` model from InsightFace.

InsightFace is used to detect faces and generate a 512-dimensional face embedding for each detected face.

The embedding is a numerical representation of facial features. Instead of directly comparing face images, the system compares these embedding vectors.

## Enrollment

To enroll a person, the system:

1. Takes an image containing one face.
2. Detects the face.
3. Generates its embedding.
4. Normalizes the embedding.
5. Stores the embedding along with the person's name.

The face embedding database is stored locally.

Example:

```text
Veeksha -> Face Embedding
```

## Face Identification

For identification, the system takes a new image and performs the same face detection and embedding process.

The new embedding is compared with the enrolled embeddings using cosine similarity.

The person with the highest similarity is selected as the best match.

A threshold is then applied:

```text
Similarity >= Threshold  -> Recognized
Similarity < Threshold   -> Unknown
```

The current threshold used in the project is:

```text
0.45
```

The threshold can also be changed from the Streamlit interface.

The threshold is a configurable development value and would need to be tuned using a larger validation dataset for production use.

## Unknown Face Rejection

The system does not automatically assign every detected face to an enrolled person.

If the highest similarity score is below the threshold, the result is:

```text
Unknown
```

During testing, an unenrolled face produced:

```text
Similarity Score: 0.0100
Threshold: 0.45
Result: Unknown
```

This demonstrates that the system can reject a face that does not sufficiently match the enrolled face database.

## Evaluation

A small development test was performed using:

* An enrolled image of Veeksha
* A different image of Veeksha
* An image of an unenrolled person

The observed results were:

| Test                         | Expected | Predicted | Similarity | Result |
| ---------------------------- | -------- | --------- | ---------: | ------ |
| Known Face - Enrolled Image  | Veeksha  | Veeksha   |     1.0000 | PASS   |
| Known Face - Different Image | Veeksha  | Veeksha   |     0.6412 | PASS   |
| Unknown Face                 | Unknown  | Unknown   |     0.0100 | PASS   |

All three test cases passed during the development testing.

These results are based on a small test set and should not be considered a general accuracy measurement for a larger real-world dataset.

## Streamlit Interface

The project includes a Streamlit interface where a user can:

* Upload a face image
* Adjust the recognition threshold
* Detect faces in the uploaded image
* View the predicted identity
* View the similarity score
* See `Unknown` when the similarity is below the threshold

To start the application:

```bash
python -m streamlit run app.py
```

## Project Structure

```text
CodeNimbus_Face_Recognition/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── src/
    ├── enroll.py
    ├── identify.py
    ├── evaluate.py
    ├── test_face.py
    └── test_insightface.py
```

## Installation

Python 3.12 was used during development.

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment on Windows

```bash
venv\Scripts\activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Test Face Detection and Embedding

```bash
python src/test_face.py
```

This checks whether a face can be detected and whether a face embedding can be generated.

### 2. Enroll a Person

```bash
python src/enroll.py
```

The program asks for the person's name and image path.

Example:

```text
Enter person's name: Veeksha
Enter image path: data/faces/test.jpeg
```

The face embedding is then stored in the local database.

### 3. Identify a Face

```bash
python src/identify.py
```

Enter the image path when prompted.

Example:

```text
data/faces/test.jpeg
```

or:

```text
data/faces/unknown.jpeg
```

### 4. Run Evaluation

```bash
python src/evaluate.py
```

### 5. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

## Failure Cases

The current implementation can have problems in situations such as:

1. No face is detected in the image.
2. More than one face is present during enrollment.
3. The face is too small in the image.
4. Poor lighting affects face detection or recognition.
5. The face is partially covered.
6. The input face has a significantly different pose or appearance from the enrollment image.
7. A threshold that is too low may increase incorrect matches.
8. A threshold that is too high may reject genuine matches.

## Limitations

This is a small prototype and not a production-ready biometric identification system.

The current database stores one embedding for each enrolled person. Recognition performance can change depending on lighting, pose, image quality, facial appearance, and other conditions.

The evaluation dataset is also very small, so the current test results should not be considered representative of overall system accuracy.

## Possible Improvements

Some improvements that can be made in the future are:

1. Store multiple enrollment images for each person.
2. Use different poses and lighting conditions during enrollment.
3. Create a larger known and unknown test dataset.
4. Tune the similarity threshold using validation data.
5. Report additional evaluation metrics such as false acceptance and false rejection.
6. Improve the user interface.
7. Add secure storage and access control for biometric data.
8. Add appropriate consent and privacy handling before using the system with real users.

## Privacy

Face embeddings are biometric information and should be handled carefully.

The project is intended as a development prototype. For real-world use, the system would require appropriate user consent, secure storage, access control, and suitable data protection measures.

## Author

**Veeksha K R**

Developed as part of the **Code Nimbus Solutions AI/ML Internship Assignment**.
