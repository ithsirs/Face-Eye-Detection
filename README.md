
# Real-Time Face & Eye Detection System using OpenCV (Haarcascade + DNN)

This project presents a robust, real-time face and eye detection system built using **OpenCV** in Python. It implements three detection pipelines:
1. **Image Dataset-based Detection (Haarcascade)**
2. **Real-Time Webcam Detection (Haarcascade)**
3. **Enhanced Real-Time Detection (DNN + Haarcascade)**

Detection results are automatically annotated and saved, with FPS and face counts displayed in real-time.

---

## 📁 Project Structure

```
.
├── face_eye_detection.py           # Detect faces & eyes from static images (Haarcascade)
├── main.py                         # Real-time face & eye detection using Haarcascade
├── main_dnn.py                     # Real-time face & eye detection using DNN + Haarcascade
├── haarcascades/                   # Haarcascade XML files
├── models/                         # DNN model files (Caffe: .prototxt & .caffemodel)
├── detected_faces_images/         # Output from face_eye_detection.py
├── real_time_detected_faces/      # Output from main.py
├── dnn_detected_faces/            # Output from main_dnn.py
├── face_detected_output.mp4v      # Video output from main.py
└── face_detected_dnn_output.mp4v  # Video output from main_dnn.py
```

---

## ✅ Features

| Feature                            | Haarcascade        | DNN (Enhanced)       |
|------------------------------------|--------------------|----------------------|
| Static Image Detection             | ✅                 | ❌                   |
| Real-Time Webcam Detection         | ✅                 | ✅                   |
| Eye Detection                      | ✅                 | ✅ (Haarcascade)     |
| Detection Accuracy                 | Moderate           | High                 |
| Tilt/Angle Face Support            | Limited            | Excellent            |
| FPS Display                        | ✅                 | ✅                   |
| Face Count Overlay                 | ✅                 | ✅                   |
| Output Image + Video Saving        | ✅                 | ✅                   |

---

## How to Run

### 1. Detect Faces and Eyes in Image Dataset
```bash
python face_eye_detection.py
```
- Input: Images in `images/` directory
- Output: Annotated images in `detected_faces_images/`

---

### 2. Real-Time Detection (Haarcascade)
```bash
python main.py
```
- Input: Live webcam feed
- Output: Video `face_detected_output.mp4v`, snapshots in `real_time_detected_faces/`

---

### 3. Enhanced Real-Time Detection (DNN)
```bash
python main_dnn.py
```
- Uses OpenCV DNN with ResNet-10 SSD model
- Output: `face_detected_dnn_output.mp4v`, snapshots in `dnn_detected_faces/`

Make sure these model files exist:
- `models/res10_300x300_ssd_iter_140000.caffemodel`
- `models/deploy.prototxt`

---

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

Install dependencies:
```bash
pip install opencv-python numpy
```

---

## Model Files

If missing, download from OpenCV GitHub:
- [Caffe Model (.caffemodel)](https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)
- [Prototxt File (deploy.prototxt)](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)

Place both in the `models/` directory.

---

## Sample Output

[20250726_194501_224] Faces detected: 1
[20250726_194502_125] Faces detected: 2


## Notes

- Press `q` to quit webcam stream in real-time modes.
- Images are saved with timestamps when faces are detected.
- The DNN version significantly improves the detection of tilted or occluded faces.

---

## Future Work

- Integrate face recognition (e.g., FaceNet or DeepFace)
- Add emotion or expression recognition
- Deploy to edge devices (Raspberry Pi, Jetson Nano)
- Build a real-time dashboard for tracking and alerts

---

