# Real-Time Facial Emotion Detector

A beginner-friendly, real-time facial **expression** classifier built with Python, OpenCV, and a pretrained FER (Facial Expression Recognition) model.  
The application runs **entirely locally** using your webcam and does not store or transmit any data.

---

## Features

- Real-time webcam facial expression detection  
- Classifies expressions such as happy, sad, angry, surprised, and neutral  
- Face bounding box with confidence score  
- On-screen emotion probability breakdown  
- Privacy-first: all processing happens locally  
- Frame skipping for smoother performance  

---

## Tech Stack

- **Language:** Python 3.11  
- **Computer Vision:** OpenCV  
- **Emotion Recognition:** `fer` (Facial Expression Recognition)  
- **Face Detection:** MTCNN (via FER)  
- **ML Backend:** TensorFlow (pretrained model)  

---

## Setup

1. Create and activate your virtual environment
```
python3.11 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
3. Run emotion_cam.py

