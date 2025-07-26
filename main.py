import cv2
import numpy as np
import time
from datetime import datetime
import os

# Create folder to save images
if not os.path.exists("real_time_detected_faces"):
    os.makedirs("real_time_detected_faces")

# Load Haarcascades
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# Start webcam
cap = cv2.VideoCapture(0)

# Get frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define codec and VideoWriter object (we will write only when faces are detected)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('face_detected_output.mp4v', fourcc, 20.0, (frame_width, frame_height))

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    elapsed_time = current_time - prev_time
    prev_time = current_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    face_detected = len(faces) > 0

    # Annotate faces and eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Annotate FPS
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # If face detected:
    if face_detected:
        # Save video frame
        out.write(frame)

        # Save image with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        filename = f"real_time_detected_faces/face_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        # Print number of faces
        print(f"[{timestamp}] Faces detected: {len(faces)}")

    # Show the video
    cv2.imshow('Real-Time Face and Eye Detection', frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
