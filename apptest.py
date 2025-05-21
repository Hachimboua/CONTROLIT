import cv2 as cv
import face_recognition
import mediapipe as mp
import numpy as np
import os
import time

# Check if professor.jpg exists
if not os.path.exists("professor.jpg"):
    print("Error: professor.jpg not found. Please add a reference image of the professor.")
    exit()

# Load professor image and encoding
try:
    prof_image = face_recognition.load_image_file("professor.jpg")
    prof_face_locations = face_recognition.face_locations(prof_image, model="hog", number_of_times_to_upsample=2)
    if not prof_face_locations:
        print("No face found in professor.jpg.")
        exit()
    prof_encoding = face_recognition.face_encodings(prof_image, prof_face_locations, num_jitters=10)[0]
    print("Professor face loaded successfully!")
except Exception as e:
    print(f"Error loading professor image: {e}")
    exit()

# Initialize webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture detection function
def detect_hand_gesture(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_dips = [6, 10, 14, 18]
    fingers_up = []
    for tip, dip in zip(finger_tips, finger_dips):
        tip_y = hand_landmarks.landmark[tip].y
        dip_y = hand_landmarks.landmark[dip].y
        fingers_up.append(tip_y < dip_y)
    if fingers_up[0] and not any(fingers_up[1:]):
        return "Index Finger Up"
    return None

# Flags and variables
debug_mode = False
use_cnn_model = False
frame_count = 0
start_time = time.time()
fps = 0
last_detection_time = 0
recognition_memory = False

print("Press 'q' to quit | 'd' = Debug | 'm' = Toggle model (HOG/CNN)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    professor_visible = False
    process_this_frame = (frame_count % 3 == 0)

    if process_this_frame or recognition_memory:
        small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)
        detection_model = "cnn" if use_cnn_model else "hog"
        time_since_detection = time.time() - last_detection_time
        upsample_times = 2 if time_since_detection > 5 else 1
        face_locations = face_recognition.face_locations(rgb_small, model=detection_model, number_of_times_to_upsample=upsample_times)
        face_encodings = []

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations, num_jitters=1)
            for i, face_encoding in enumerate(face_encodings):
                face_distance = face_recognition.face_distance([prof_encoding], face_encoding)[0]
                confidence = 1 - face_distance
                adaptive_tolerance = min(0.70, 0.65 + (time_since_detection * 0.005))
                match = face_recognition.compare_faces([prof_encoding], face_encoding, tolerance=adaptive_tolerance)

                if match[0]:
                    professor_visible = True
                    last_detection_time = time.time()
                    recognition_memory = True
                    top, right, bottom, left = face_locations[i]
                    top *= 2; right *= 2; bottom *= 2; left *= 2
                    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv.putText(frame, f"Professor ({confidence:.0%})", (left, top - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    break
                elif debug_mode:
                    top, right, bottom, left = face_locations[i]
                    top *= 2; right *= 2; bottom *= 2; left *= 2
                    cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv.putText(frame, f"Unknown ({confidence:.0%})", (left, top - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if time.time() - last_detection_time > 2.0:
        recognition_memory = False

    if professor_visible:
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = pose_results.pose_landmarks.landmark
            frame_height, frame_width, _ = frame.shape
            wrist_indices = [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]

            for wrist_index in wrist_indices:
                wrist = landmarks[wrist_index]
                if wrist.visibility < 0.5:
                    continue

                x = int(wrist.x * frame_width)
                y = int(wrist.y * frame_height)
                box_size = 150
                offset = 40
                x1 = max(x - box_size // 2, 0)
                y1 = max(y - box_size // 2 - offset, 0)
                x2 = min(x + box_size // 2, frame_width)
                y2 = min(y + box_size // 2 - offset, frame_height)

                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                hand_region = frame[y1:y2, x1:x2].copy()

                if hand_region.size > 0:
                    hand_rgb = cv.cvtColor(hand_region, cv.COLOR_BGR2RGB)
                    hand_results = hands.process(hand_rgb)

                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            region_height, region_width, _ = hand_region.shape
                            for lm in hand_landmarks.landmark:
                                lm_x = int(lm.x * region_width)
                                lm_y = int(lm.y * region_height)
                                cv.circle(hand_region, (lm_x, lm_y), 3, (0, 255, 255), -1)

                            mp_drawing.draw_landmarks(
                                hand_region, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                            )

                            gesture = detect_hand_gesture(hand_landmarks)
                            if gesture:
                                cv.putText(frame, f"Gesture: {gesture}", (x1, y1 - 40),
                                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                        frame[y1:y2, x1:x2] = hand_region

    # Display status
    if professor_visible or recognition_memory:
        status_color = (0, 255, 0)
        status_text = "Professor Detected - Tracking Active"
    else:
        time_diff = time.time() - last_detection_time
        status_color = (0, 165, 255) if time_diff < 5 else (0, 0, 255)
        status_text = "Professor Lost - Searching..." if time_diff < 5 else "Waiting for Professor..."

    cv.putText(frame, status_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Debug info
    if debug_mode:
        cv.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        model_name = "CNN" if use_cnn_model else "HOG"
        cv.putText(frame, f"Model: {model_name}", (10, frame.shape[0] - 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(frame, f"Last detection: {time.time() - last_detection_time:.1f}s ago",
                   (10, frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv.imshow("Professor Gesture Control", frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    elif key == ord('m'):
        use_cnn_model = not use_cnn_model
        print(f"Using {'CNN' if use_cnn_model else 'HOG'} detection model")

# Cleanup
cap.release()
cv.destroyAllWindows()
