# Face Recognition Identification System
A simple face recognition system built as part of the Code Nimbus Solutions AI/ML internship assignment.
The system can enroll a person's face, generate a face embedding, store it locally, and identify a new face by comparing it with the enrolled faces. If the similarity is below the selected threshold, the system returns `Unknown`.

## What the Project Does
The project has two main parts:
- Face enrollment
- Face identification
During enrollment, the system detects the face in an image and saves its embedding in a local JSON database.
During identification, a new face is detected and its embedding is compared with the stored embeddings using cosine similarity.

## Technologies Used
- Python
- InsightFace
- ONNX Runtime
- OpenCV
- NumPy
- Scikit-learn
- Streamlit

## How It Works
The basic flow is:
Input Image
     |
     v
Face Detection
     |
     v
Face Embedding
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
Similarity >= Threshold   Similarity < Threshold
     |                      |
     v                      v
  Person Name             Unknown

## Face Detection and Embeddings
The project uses the `buffalo_l` model from InsightFace.
InsightFace is used to detect faces and generate a 512-dimensional embedding for each detected face.
The embedding is a numerical representation of the facial features. Instead of storing the face image for matching, the system compares these embedding vectors.

## Enrollment
To enroll a person, the system:
1. Takes an image containing one face.
2. Detects the face.
3. Generates its embedding.
4. Normalizes the embedding.
5. Stores the embedding along with the person's name.

The current local database is stored in:
data/database/embeddings.json
Example:
Veeksha -> face embedding


## Face Identification
For identification, the system takes a new image and performs the same face detection and embedding process.
The new embedding is compared with the enrolled embeddings using cosine similarity.
The person with the highest similarity is selected as the best match.
A threshold is then applied:
Similarity >= threshold  -> Recognized
Similarity < threshold   -> Unknown
The current threshold used in the project is:
0.45
The threshold can also be changed from the Streamlit interface.
The threshold is currently a configurable development value and would need to be tuned using a larger validation dataset for a production system.

## Unknown Face Rejection
The system does not automatically assign every detected face to an enrolled person.
If the highest similarity score is below the threshold, the result is:
Unknown
During testing, an unenrolled face produced:
Similarity Score: 0.1616
Threshold: 0.45
Result: Unknown
This allows the system to reject faces that do not sufficiently match the enrolled database.

## Evaluation
A small test was performed using:
* One image of the enrolled person
* One image of a person who was not enrolled

The results were:
| Test         | Expected | Predicted | Similarity | Result |
| ------------ | -------- | --------- | ---------: | ------ |
| Known Face   | Veeksha  | Veeksha   |     1.0000 | PASS   |
| Unknown Face | Unknown  | Unknown   |     0.1616 | PASS   |
Both test cases passed:
2 / 2 test cases passed
The observed test accuracy for this small development test was:
100%
This is only a small development test and is not intended to represent the accuracy of the system on a larger real-world dataset.

## Streamlit Interface
The project includes a Streamlit interface where a user can:
* Upload a face image
* Adjust the recognition threshold
* Detect faces in the uploaded image
* View the predicted identity
* View the similarity score
* See `Unknown` when the similarity is below the threshold

To start the application:
python -m streamlit run app.py

## Project Structure
CodeNimbus_Face_Recognition/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── database/
│   │   └── embeddings.json
│   │
│   └── faces/
│
├── models/
│
├── src/
│   ├── enroll.py
│   ├── identify.py
│   ├── evaluate.py
│   ├── test_face.py
│   └── test_insightface.py
│
└── venv/

## Installation
Python 3.12 was used during development.
First, create a virtual environment:
python -m venv venv
Activate the virtual environment on Windows:
venv\Scripts\activate
Install the required packages:
pip install -r requirements.txt

## Running the Project

### 1. Test Face Detection and Embedding
python src/test_face.py
This checks whether a face can be detected and whether a 512-dimensional embedding can be generated.

### 2. Enroll a Person
python src/enroll.py
The program asks for the person's name and image path.
Example:
Enter person's name: Veeksha
Enter image path: data/faces/test.jpeg
The face embedding is then saved in:
data/database/embeddings.json

### 3. Identify a Face
python src/identify.py
Enter the image path when prompted.
For example:
data/faces/test.jpeg
or:
data/faces/unknown.jpeg

### 4. Run Evaluation
python src/evaluate.py

### 5. Run the Streamlit Application
python -m streamlit run app.py

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
The current database stores one embedding for each enrolled person. Recognition performance can change depending on lighting, pose, image quality and other factors.
The evaluation dataset is also very small, so the current test result should not be considered a general accuracy measurement.

## Possible Improvements
Some improvements that can be made in the future are:
1. Store multiple enrollment images for each person.
2. Use different poses and lighting conditions during enrollment.
3. Create a larger known and unknown test dataset.
4. Tune the similarity threshold using validation data.
5. Report additional evaluation metrics such as false acceptance and false rejection.
6. Improve the user interface.
7. Add secure storage and access control for biometric data.
8. Add proper consent and privacy handling before using the system with real users.

## Privacy
Face embeddings are biometric information and should be handled carefully.
The current project keeps the face images and local embedding database outside the public GitHub repository using `.gitignore`.
For real-world use, the system would require appropriate user consent, secure storage and suitable data protection measures.

## Author
**Veeksha K R**
Developed as part of the Code Nimbus Solutions AI/ML internship assignment.
