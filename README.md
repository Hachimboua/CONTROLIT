# 🎛️ CONTROLIT: Hand & Voice-Controlled Slideshow System

**CONTROLIT** is an AI-powered system enabling professors to seamlessly control slideshow presentations using **hand gestures** and **voice commands**. It combines **face recognition** (for security), **gesture tracking** (via MediaPipe), and **speech recognition** (via Vosk) for a fully hands-free experience.

---

## 🔑 Key Features

### 👨‍🏫 Secure Professor Authentication

* Uses `face_recognition` to ensure only the authorized professor can control the system.

### ✋ Smart Gesture Control

* **Right-hand only** gesture recognition using MediaPipe Hands and a custom PyTorch classifier.
* Processes hands within an **adaptive Region of Interest (ROI)** around the wrist.
* Gesture **cooldown logic** prevents repeated accidental commands.
* **Single-hand mode** for stable control (no multi-hand confusion).
* Built-in **keypoint data logger** for custom gesture collection and model retraining.

### 🎤 Voice Command Integration

* Uses **Vosk** for real-time offline speech recognition.
* Recognizes simple commands:

  * `"next"`, `"forward"`, `"right"` → Next Slide
  * `"previous"`, `"back"`, `"left"` → Previous Slide
  * `"quit"`, `"exit"` → Exit App
* Automatically activated via a specific gesture (e.g., "Activate Voice Mode").

### 🧠 Optimized Pose & Vision Processing

* **MediaPipe Pose** runs in a region around the professor’s face for efficiency.
* Downscaled face recognition for speed.
* **Dynamic re-centering** of the ROI if tracking drifts.

### 🐞 Developer-Friendly Debugging

* Toggleable debug overlays show ROI, FPS, gesture class, and voice status.
* Terminal logging for all recognized actions and system states.

---

## 💻 Demo

### 👉 Gesture Example: "Next Slide"

![Next Slide Gesture](/images/next_up.png)

---

## 🛠️ Installation & Setup

### ✅ Requirements

Ensure Python 3.8+ is installed.

Install required packages:

```bash
pip install opencv-python mediapipe face_recognition pyautogui numpy torch torchvision pytorch-lightning torchmetrics scikit-learn sounddevice vosk
```

Additionally, download and extract a **Vosk speech model** (e.g., [vosk-model-small-en-us-0.15](https://alphacephei.com/vosk/models)) into your project folder:

```
/vosk-model-small-en-us-0.15/
```

---

## 📁 Project Structure

```
controlit/
├── model/
│   ├── keypoint_classifier/
│   │   ├── keypoint_classifier_weights.pth
│   │   ├── keypoint_classifier_label.csv
│   │   └── keypoint.csv
├── professor.jpg         # Authorized professor face image
├── vosk-model-small-en-us-0.15/  # Speech model directory
├── main.py               # Main script
├── README.md
```

---

## 🎯 Supported Gestures

| Gesture Label | Action                 |
| ------------- | ---------------------- |
| `0`           | Activate Voice Mode    |
| `1`           | Start Slideshow (`F5`) |
| `2`           | Pause/Resume (`Space`) |
| `3`           | Previous Slide (`←`)   |
| *(...)*       | Extendable             |

You can **collect new gestures** by entering "Logging Mode" (`k` key) and saving labeled data with number keys (`0`–`9`).

---

## 🗣️ Voice Commands

After performing the “Activate Voice Mode” gesture, the system will listen for voice commands for a few seconds:

| Spoken Command     | Action               |
| ------------------ | -------------------- |
| `next`, `forward`  | Next Slide (`→`)     |
| `previous`, `back` | Previous Slide (`←`) |
| `quit`, `exit`     | Exit App             |

---

## 🎛️ Controls

| Key          | Action                              |
| ------------ | ----------------------------------- |
| `Esc` or `q` | Quit application                    |
| `k`          | Start gesture logging               |
| `n`          | Return to normal mode               |
| `d`          | Toggle debug overlays               |
| `0-9`        | Log gesture (while in logging mode) |

---

## 🧬 Model Training (Optional)

If you wish to retrain the gesture model:

1. Use logging mode to generate labeled `keypoint.csv` data.
2. Train the classifier using PyTorch and save weights as `keypoint_classifier_weights.pth`.
3. Update `keypoint_classifier_label.csv` to match your new classes.

---

## ⚠️ Known Limitations

* Currently optimized for **one user** in front of the camera.
* Voice mode deactivates after 5 seconds of inactivity.
* Assumes a **static background** for reliable detection.

---

## 👥 Credits

Built with:

* [MediaPipe](https://google.github.io/mediapipe/)
* [face\_recognition](https://github.com/ageitgey/face_recognition)
* [Vosk](https://alphacephei.com/vosk/)
* [PyTorch](https://pytorch.org/)
* [OpenCV](https://opencv.org/)
