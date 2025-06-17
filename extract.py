import cv2
import numpy as np
from paddleocr import PaddleOCR
import re

# === Initialize OCR ===
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    drop_score=0.5,
    use_space_char=True,
)

# === Rectangle guide size (Pakistan CNIC size ~ 450x300 px) ===
RECT_WIDTH = 400
RECT_HEIGHT = 250

# === Initialize Camera ===
cap = cv2.VideoCapture(1)  # Adjust index if wrong camera
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# === Get camera frame size ===
ret, frame = cap.read()
H, W = frame.shape[:2]
center_x, center_y = W // 2, H // 2
rect_top_left = (center_x - RECT_WIDTH // 2, center_y - RECT_HEIGHT // 2)
rect_bottom_right = (center_x + RECT_WIDTH // 2, center_y + RECT_HEIGHT // 2)

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
    """Basic text cleaner"""
    text = text.replace('\n', ' ').strip()
    return text

def map_fields(ocr_results):
    """Map OCR results to CNIC fields"""
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

            # Heuristic mappings
            if re.match(r"^[A-Za-z]+\s[A-Za-z]+$", text) and fields["Name"] is None:
                fields["Name"] = text

            elif re.search(r"(Father|ather|Fther)", text, re.IGNORECASE):
                continue  # Skip label

            elif re.match(r"^[A-Za-z]+\s[A-Za-z]+$", text) and fields["Father Name"] is None and fields["Name"] is not None:
                fields["Father Name"] = text

            elif text in ["M", "F"] and fields["Gender"] is None:
                fields["Gender"] = text

            elif text.lower() in ["pakistan"] and fields["Country of Stay"] is None:
                fields["Country of Stay"] = text

            elif re.match(r"^\d{5}-\d{7}-\d$", text) and fields["Identity Number"] is None:
                fields["Identity Number"] = text

            elif re.match(r"\d{2}\.\d{2}\.\d{4}", text):
                # Decide based on context
                if fields["Date of Birth"] is None:
                    fields["Date of Birth"] = text
                elif fields["Date of Issue"] is None:
                    fields["Date of Issue"] = text
                elif fields["Date of Expiry"] is None:
                    fields["Date of Expiry"] = text

    return fields

def run_full_image_ocr(image):
    """Run OCR on cropped image and map fields"""
    ocr_result = ocr.ocr(image, cls=True)

    annotated_image = image.copy()

    print("\n--- OCR Results ---")
    for line in ocr_result:
        if line is None or len(line) == 0:
            continue
        for box, (text, confidence) in line:
            box = np.array(box).astype(int)
            cv2.polylines(annotated_image, [box], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(annotated_image, text, tuple(box[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            coords = box.tolist()
            print(f"Text: '{text}' | Box: {coords} | Score: {confidence:.2f}")

    print("-------------------")

    # === FIELD MAPPING ===
    extracted_fields = map_fields(ocr_result)
    print("\n=== Extracted Fields ===")
    for key, value in extracted_fields.items():
        print(f"{key}: {value}")
    print("========================")

    cv2.imshow("OCR Result", annotated_image)
    print("[INFO] Close the window or press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
            run_full_image_ocr(cropped_card)
        else:
            print("[WARNING] Card not fully inside the rectangle. Adjust position before capturing.")

# === Clean up ===
cap.release()
cv2.destroyAllWindows()

