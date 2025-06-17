import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
import sqlite3

# === Initialize OCR ===
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    drop_score=0.5,
    use_space_char=True,
)

# === Load Face Detector ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Constants ===
RECT_WIDTH = 400
RECT_HEIGHT = 250
FACE_SAVE_PATH = ""
DATABASE_NAME = ""
TABLE_NAME = ""

# === Initialize Camera ===
cap = cv2.VideoCapture(1)  # Adjust if wrong camera index
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get camera frame size
ret, frame = cap.read()
H, W = frame.shape[:2]
center_x, center_y = W // 2, H // 2
rect_top_left = (center_x - RECT_WIDTH // 2, center_y - RECT_HEIGHT // 2)
rect_bottom_right = (center_x + RECT_WIDTH // 2, center_y + RECT_HEIGHT // 2)

# === Initialize SQLite DB ===
conn = sqlite3.connect(DATABASE_NAME)
cursor = conn.cursor()

cursor.execute(f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    "Name" TEXT,
    "Father_Name" TEXT,
    "Gender" TEXT,
    "Country_of_Stay" TEXT,
    "Identity_Number" TEXT PRIMARY KEY,
    "Date_of_Birth" TEXT,
    "Date_of_Issue" TEXT,
    "Date_of_Expiry" TEXT
)
""")
conn.commit()

# === Helper Functions ===
def is_card_inside(frame, rect_top_left, rect_bottom_right):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    card_contour = None
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                card_contour = approx
    if card_contour is not None:
        all_inside = True
        x1, y1 = rect_top_left
        x2, y2 = rect_bottom_right
        for pt in card_contour:
            px, py = pt[0]
            if not (x1 + 10 < px < x2 - 10 and y1 + 10 < py < y2 - 10):
                all_inside = False
                break
        return all_inside, card_contour
    return False, None

def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm

def crop_to_rectangle(image, rect_top_left, rect_bottom_right):
    x1, y1 = rect_top_left
    x2, y2 = rect_bottom_right
    return image[y1:y2, x1:x2]

def clean_text(text):
    return text.replace('\n', ' ').strip()

def map_fields(ocr_results):
    fields = {
        "Name": None,
        "Father Name": None,
        "Gender": None,
        "Country of Stay": None,
        "Identity Number": None,
        "Date of Birth": None,
        "Date of Issue": None,
        "Date of Expiry": None,
    }

    for line in ocr_results:
        if line is None or len(line) == 0:
            continue
        for box, (text, confidence) in line:
            text = clean_text(text)
            if re.match(r"^[A-Za-z]+\s[A-Za-z]+$", text) and fields["Name"] is None:
                fields["Name"] = text
            elif re.search(r"(Father|ather|Fther)", text, re.IGNORECASE):
                continue
            elif re.match(r"^[A-Za-z]+\s[A-Za-z]+$", text) and fields["Father Name"] is None and fields["Name"] is not None:
                fields["Father Name"] = text
            elif text in ["M", "F"] and fields["Gender"] is None:
                fields["Gender"] = text
            elif text.lower() == "pakistan" and fields["Country of Stay"] is None:
                fields["Country of Stay"] = text
            elif re.match(r"^\d{5}-\d{7}-\d$", text) and fields["Identity Number"] is None:
                fields["Identity Number"] = text
            elif re.match(r"\d{2}\.\d{2}\.\d{4}", text):
                if fields["Date of Birth"] is None:
                    fields["Date of Birth"] = text
                elif fields["Date of Issue"] is None:
                    fields["Date of Issue"] = text
                elif fields["Date of Expiry"] is None:
                    fields["Date of Expiry"] = text

    return fields

def extract_and_save_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("[INFO] No face detected in the captured CNIC region.")
        return False

    # Save first detected face
    (x, y, w, h) = faces[0]
    cropped_face = image[y:y+h, x:x+w]
    cv2.imwrite(FACE_SAVE_PATH, cropped_face)
    print(f"[INFO] Face extracted and saved as '{FACE_SAVE_PATH}'")

    # Optional: show the extracted face
    cv2.imshow("Extracted Face", cropped_face)
    cv2.waitKey(500)
    cv2.destroyWindow("Extracted Face")

    return True

def run_full_image_ocr_and_save(image):
    ocr_result = ocr.ocr(image, cls=True)

    annotated_image = image.copy()
    for line in ocr_result:
        if line is None or len(line) == 0:
            continue
        for box, (text, confidence) in line:
            box = np.array(box).astype(int)
            cv2.polylines(annotated_image, [box], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(annotated_image, text, tuple(box[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    extracted_fields = map_fields(ocr_result)
    print("\n=== Extracted Fields ===")
    for key, value in extracted_fields.items():
        print(f"{key}: {value}")
    print("========================")

    # === SAVE TO DATABASE ===
    if any(value is None for value in extracted_fields.values()):
        print("[WARNING] Some fields are missing. Skipping database insertion.")
    else:
        try:
            cursor.execute(f"""
                INSERT INTO {TABLE_NAME} VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                extracted_fields["Name"],
                extracted_fields["Father Name"],
                extracted_fields["Gender"],
                extracted_fields["Country of Stay"],
                extracted_fields["Identity Number"],
                extracted_fields["Date of Birth"],
                extracted_fields["Date of Issue"],
                extracted_fields["Date of Expiry"]
            ))
            conn.commit()
            print("[INFO] Data successfully saved into database.")
        except sqlite3.IntegrityError:
            print(
                f"[WARNING] CNIC {extracted_fields['Identity Number']} already exists in the database. Skipping insertion.")

    cv2.imshow("OCR Result", annotated_image)
    print("[INFO] Close the window or press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow("OCR Result")

# === MAIN LOOP ===
print("[INFO] Starting camera feed. Press 'c' to capture, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    inside, card_contour = is_card_inside(frame, rect_top_left, rect_bottom_right)
    sharpness = calculate_sharpness(frame)

    if inside:
        color = (0, 255, 0)
        status_text = "Card Detected - Press 'c' to capture"
    else:
        color = (0, 0, 255)
        status_text = "Align CNIC fully inside box"

    cv2.rectangle(frame, rect_top_left, rect_bottom_right, color, 2)
    cv2.putText(frame, status_text, (rect_top_left[0], rect_top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    sharpness_text = f"Sharpness: {sharpness:.2f}"
    cv2.putText(frame, sharpness_text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("CNIC Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("[INFO] Quitting...")
        break
    elif key == ord('c'):
        if inside:
            print("[INFO] Card detected inside the guide rectangle.")
            print(f"[INFO] Sharpness Score: {sharpness:.2f}")
            if sharpness < 150:
                print("[WARNING] Image is too blurry (sharpness < 150). Please try again.")
                continue

            cropped_card = crop_to_rectangle(frame, rect_top_left, rect_bottom_right)

            # === STEP 1: FACE EXTRACTION ===
            face_found = extract_and_save_face(cropped_card)

            # === STEP 2: TEXT EXTRACTION + SAVE ===
            run_full_image_ocr_and_save(cropped_card)

        else:
            print("[WARNING] Card not fully inside the rectangle. Adjust position before capturing.")

cap.release()
cv2.destroyAllWindows()
conn.close()
