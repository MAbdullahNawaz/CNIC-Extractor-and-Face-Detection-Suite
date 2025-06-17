CNIC-Extractor-and-Face-Detection-Suite
This project focuses on extracting relevant personal information from CNIC (Computerized National Identity Card) images using Python, OpenCV, and PaddleOCR. It is specifically designed to:

Capture CNIC images via webcam.

Detect the card's position and quality.

Extract important identity fields like Name, Father’s Name, CNIC Number, Gender, Dates, etc.

Detect and crop faces from the CNIC image.

Project Structure & File-wise Explanation
extract.py
This is the main driver file of the project.
It does the following:

Opens the system's webcam and displays a guide rectangle for positioning the CNIC.

Detects whether the CNIC is properly aligned inside the rectangle.

Checks the image sharpness to ensure readability.

On pressing c:

Captures the CNIC image.

Crops it to the rectangle.

Applies PaddleOCR to extract text from the image.

Maps the extracted text to CNIC fields like:

Name

Father’s Name

CNIC Number

Date of Birth

Date of Issue

Date of Expiry

Gender

Country of Stay

Displays the recognized fields and annotated CNIC image.

Provides a simple, interactive loop to capture or quit.

Image.py
This file focuses on face detection from the CNIC image.
It does the following:

Reads a saved CNIC image from your system.

Detects faces using OpenCV's pre-trained Haar cascade model.

Crops the detected face.

Displays and saves the cropped face image.

MERGEDdb.py
This file performs the same face detection logic as Image.py but is intended to be expanded for:

Database operations (in the next development phase).

Merging and storing the extracted data.

Currently, it's structured for face detection and saving cropped images, but it can be improved to:

Save extracted text into an SQLite database.

Merge face images and CNIC data for complete user profiles.

.gitignore
The .gitignore file is properly configured to:

Ignore PyCharm-specific files (.idea/).

Ignore the virtual environment folder (.venv/).

Ignore the local PaddleOCR directory (PaddleOCR/).

Ignore Python cache files (__pycache__/, *.pyc).

This ensures that unnecessary files are not pushed to the remote repository.

How to Run
Clone the Repository:

bash
Copy
Edit
git clone <repository-url>
Setup Virtual Environment:

bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate  # For Linux/Mac
.venv\Scripts\activate     # For Windows
Install Dependencies:

bash
Copy
Edit
pip install paddleocr opencv-python numpy
Run the Main File:

bash
Copy
Edit
python extract.py
Key Functionalities
Real-time CNIC detection via webcam.

Image sharpness validation.

Accurate OCR using PaddleOCR.

Clean mapping of extracted CNIC fields.

Face detection and cropping.

Organized and extendable code structure.

