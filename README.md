<h1>NIC Extraction Project</h1>

<p>This project focuses on extracting relevant personal information from CNIC (Computerized National Identity Card) images using <strong>Python, OpenCV, and PaddleOCR</strong>.</p>

<p><strong>Core Features:</strong></p>
<ul>
    <li>Capture CNIC images via webcam.</li>
    <li>Detect card position and image quality.</li>
    <li>Extract important identity fields like Name, Father’s Name, CNIC Number, Gender, Dates, etc.</li>
    <li>Detect and crop faces from the CNIC image.</li>
</ul>

<hr>

<h2>Project Structure & File-wise Explanation</h2>

<h3>extract.py</h3>
<p>This is the <strong>main driver file</strong> of the project.</p>
<p><strong>Responsibilities:</strong></p>
<ul>
    <li>Access webcam and display a guide rectangle for CNIC placement.</li>
    <li>Check CNIC alignment and image sharpness.</li>
    <li>Capture and crop the CNIC image on pressing <code>c</code>.</li>
    <li>Extract text using PaddleOCR.</li>
    <li>Map extracted text to CNIC fields:
        <ul>
            <li>Name</li>
            <li>Father’s Name</li>
            <li>CNIC Number</li>
            <li>Date of Birth</li>
            <li>Date of Issue</li>
            <li>Date of Expiry</li>
            <li>Gender</li>
            <li>Country of Stay</li>
        </ul>
    </li>
    <li>Display annotated CNIC with extracted fields.</li>
</ul>

<h3>Image.py</h3>
<p>This file handles <strong>face detection</strong> from CNIC images.</p>
<p><strong>Responsibilities:</strong></p>
<ul>
    <li>Read saved CNIC images from the system.</li>
    <li>Detect faces using OpenCV’s Haar cascade model.</li>
    <li>Crop and save the detected face image.</li>
</ul>

<h3>MERGEDdb.py</h3>
<p>This file is a <strong>future extension</strong> for face detection and potential database operations.</p>
<p><strong>Responsibilities:</strong></p>
<ul>
    <li>Currently performs the same face detection as Image.py.</li>
    <li>Can be expanded to:
        <ul>
            <li>Store extracted CNIC data in a database.</li>
            <li>Merge user profiles with cropped face images.</li>
        </ul>
    </li>
</ul>

<h3>.gitignore</h3>
<p>This file ensures unnecessary files are not pushed to the remote repository.</p>
<ul>
    <li>Ignores PyCharm files: <code>.idea/</code></li>
    <li>Ignores virtual environments: <code>.venv/</code></li>
    <li>Ignores local PaddleOCR folder: <code>PaddleOCR/</code></li>
    <li>Ignores Python cache files: <code>__pycache__/</code> and <code>*.pyc</code></li>
</ul>

<hr>

<h2>How to Run</h2>

<ol>
    <li><strong>Clone the Repository:</strong>
        <pre><code>git clone &lt;repository-url&gt;</code></pre>
    </li>
    <li><strong>Setup Virtual Environment:</strong>
        <pre><code>python -m venv .venv
source .venv/bin/activate  # For Linux/Mac
.venv\Scripts\activate     # For Windows</code></pre>
    </li>
    <li><strong>Install Dependencies:</strong>
        <pre><code>pip install paddleocr opencv-python numpy</code></pre>
    </li>
    <li><strong>Run the Main File:</strong>
        <pre><code>python extract.py</code></pre>
    </li>
</ol>

<hr>

<h2>Key Functionalities</h2>
<ul>
    <li>Real-time CNIC detection via webcam.</li>
    <li>Image sharpness validation.</li>
    <li>Accurate OCR using PaddleOCR.</li>
    <li>Mapping extracted text to proper CNIC fields.</li>
    <li>Face detection and cropping.</li>
    <li>Clean, organized, and extendable project structure.</li>
</ul>

<hr>

<h2>Author</h2>
<p><strong>M Abdullah</strong><br>
Associate AI/ML Engineer</p>
