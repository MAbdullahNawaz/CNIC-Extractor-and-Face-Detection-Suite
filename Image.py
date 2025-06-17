import cv2

# Upload your Image here
image = cv2.imread(r"")

# Load OpenCV's pretrained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Loop through faces and crop
for (x, y, w, h) in faces:
    cropped_face = image[y:y+h, x:x+w]
    cv2.imwrite("cropped_face.jpg", cropped_face)
    cv2.imshow("Face", cropped_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

