import cv2
import os
import matplotlib.pyplot as plt

# Set your dataset path and output path
dataset_path = r'D:\\Face and Eye Detection\\images\\img'
output_path = 'detected_faces_images'


# Check if dataset path exists
if not os.path.exists(dataset_path):
    print(f" Dataset path not found: {dataset_path}")
    exit()

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# List all image files
image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for filename in image_files:
    img_path = os.path.join(dataset_path, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f" Failed to load image: {filename}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )

    for i, (x, y, w, h) in enumerate(faces):
        # Draw face rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract face ROI (region of interest)
        face_roi_color = img[y:y + h, x:x + w]
        face_roi_gray = gray[y:y + h, x:x + w]

        # Detect eyes within face ROI
        eyes = eye_cascade.detectMultiScale(
            face_roi_gray, scaleFactor=1.05, minNeighbors=5, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Save cropped face with eyes
        face_filename = f"{os.path.splitext(filename)[0]}_face_{i}.jpg"
        #cv2.imwrite(os.path.join(output_path, face_filename), face_roi_color)

    # Save annotated image
    annotated_path = os.path.join(output_path, f"annotated_{filename}")
    cv2.imwrite(annotated_path, img)

    print(f" Processed {filename} - {len(faces)} face(s) found")

print(" Face and eye detection completed on all images.")
