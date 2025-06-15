# CONTROLIT: Hand Gesture Slideshow System 👨‍🏫

**CONTROLIT** offers a secure, intuitive, and real-time way for a professor to control slideshow presentations using only hand gestures. The system combines advanced computer vision techniques—**face recognition** for authentication and **MediaPipe** for pose and hand tracking—to deliver a seamless, gesture-controlled experience.

---

## 🚀 Features

- **🧑‍🏫 Professor Authentication**  
  Ensures that only the authorized professor can control the slideshow using face recognition.

- **⚙️ Optimized Pose Tracking**  
  MediaPipe Pose is applied selectively to a Region of Interest (ROI) around the professor’s face, reducing computation and preventing interference.

- **📍 Adaptive Hand ROI**  
  Hand detection focuses dynamically around the professor’s wrists, adapting to their distance from the camera.

- **⚡ Efficient Processing**  
  Face recognition runs on downscaled frames and less frequently once authentication is confirmed.

- **🎚️ Adjustable Face Tolerance**  
  Recognition strictness adjusts based on recent detection confidence history.

- **✋ Gesture Cooldown**  
  Prevents rapid repeat execution of the same command.

- **🖐️ Single-Hand Processing**  
  Explicitly processes gestures from only one hand (right arm only for better consistency).

- **🐛 Enhanced Debugging**  
  Visual debug overlays and detailed terminal output assist in system troubleshooting.

- **📊 Data Collection Mode**  
  Built-in data logger makes it easy to collect training data for new gestures.

---

## 📸 Demos & Screenshots

> **Please replace the placeholders below with actual images or GIFs**

- **Professor Authentication**  
  _Image/GIF of system recognizing the professor_

- **Gesture Control (e.g., "Next Slide")**  
  _Image/GIF of a gesture triggering the "next slide" command_

- **Debug Mode Visualization**  
  _Screenshot showing face ROI, hand ROI, and landmarks_

- **Data Collection Mode**  
  _Screenshot of logging interface and hand keypoints_

---

## 🛠️ Setup and Installation

### ✅ Prerequisites

Ensure the following are installed:

- Python 3.8+
- OpenCV
- MediaPipe
- face_recognition
- pyautogui
- NumPy
- PyTorch & torchvision
- PyTorch Lightning
- torchmetrics
- scikit-learn

Install using `pip`:

```bash
pip install opencv-python mediapipe face_recognition pyautogui numpy torch torchvision pytorch-lightning torchmetrics scikit-learn
