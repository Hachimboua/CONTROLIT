import csv
import copy
import argparse
import itertools
from collections import deque
import pyautogui
import time
import os
import torch
import sys

import cv2 as cv
import numpy as np
import mediapipe as mp
import face_recognition

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from model.keypoint_classifier.keypoint_classifier_pyt import KeyPointClassifier
from utils import CvFpsCalc

"""
Professor-Controlled Hand Gesture Slideshow System

This module implements a real-time hand gesture recognition system for slideshow control,
secured by face recognition to ensure only the designated "professor" can operate it.
It leverages MediaPipe for robust hand and pose landmark detection.

Key Features & Optimizations:
1.  **Professor Authentication:** Uses face_recognition to identify the professor.
2.  **Professor-Specific Pose Tracking:** MediaPipe Pose is applied only to a dynamically
    calculated region around the professor's detected face, preventing detection of other
    individuals' bodies and improving efficiency.
3.  **Wrist-Based Hand ROI:** Hand detection is focused on a dynamically calculated region
    around the professor's wrists, adapting to their distance from the camera.
4.  **Optimized Frame Processing:** Face recognition is done on downscaled frames and
    much less frequently once the professor is detected.
5.  **Adaptive Face Tolerance:** Adjusts recognition strictness based on recent detection history.
6.  **Gesture Cooldown:** Prevents rapid-fire actions from the same gesture.
7.  **Single Hand Processing:** Explicitly ensures only one hand's gestures are processed
    for commands, even if MediaPipe's internal output temporarily shows more.
8.  **Strict Left Arm Only Focus:** MediaPipe Pose *only* considers the anatomical left arm
    for wrist detection, ensuring only the physical left hand can trigger gestures.
9.  **Enhanced Debugging:** Includes explicit visual cues and print statements for wrist
    detection and their visibility scores to aid in troubleshooting.
10. **Performance Enhancements:** Includes MediaPipe model complexity adjustments and
    optimized landmark array processing for utility functions.
11. **Data Collection Mode:** Press 'k' to enter a mode to collect hand keypoint data for
    new gestures. Press '0' through '9' to label and save data to 'model/keypoint_classifier/keypoint.csv'.

Dependencies:
    - OpenCV (cv2)
    - MediaPipe (mediapipe)
    - face_recognition
    - PyAutoGUI (pyautogui)
    - NumPy (numpy)
    - PyTorch (torch) - if your KeyPointClassifier models use it.
"""

def get_args():
    """Parses command line arguments for the integrated system."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--width", help='Camera capture width', type=int, default=960)
    parser.add_argument("--height", help='Camera capture height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true',
                        help="MediaPipe Hands: Treat input images as a batch, ideal for static images.")
    parser.add_argument("--min_detection_confidence",
                        help='MediaPipe Hands: Minimum confidence score for hand detection to be considered successful.',
                        type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='MediaPipe Hands: Minimum confidence score for hand tracking to be considered successful.',
                        type=float, default=0.5)

    args = parser.parse_args()
    return args


def main():
    """
    Main function implementing the "Best of Both Worlds" pipeline:
    1. State Machine: Initial Face Search -> Pose Tracking -> (if fails) Re-acquire.
    2. Tracking Mechanism: Detailed, chained-ROI logic from the original script.
    """
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_hand_detection_confidence = args.min_detection_confidence
    min_hand_tracking_confidence = args.min_tracking_confidence
    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    professor_image_path = "professor.jpg"
    try:
        prof_image = face_recognition.load_image_file(professor_image_path)
        prof_face_locations = face_recognition.face_locations(prof_image, model="hog")
        prof_encoding = face_recognition.face_encodings(prof_image, prof_face_locations, num_jitters=50)[0]
    except Exception as e:
        print(f"FATAL: Could not load professor image from {professor_image_path}. Error: {e}")
        return

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode, max_num_hands=1,
        min_detection_confidence=min_hand_detection_confidence, min_tracking_confidence=min_hand_tracking_confidence,
        model_complexity=0
    )

    try:
        keypoint_classifier = KeyPointClassifier("model/keypoint_classifier/keypoint_classifier_weights.pth")
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    except Exception as e:
        print(f"FATAL: Could not load gesture models. Error: {e}")
        return
        
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    last_executed_time = 0
    gesture_cooldown = 1.5
    last_executed_gesture = -1

    current_mode = 'SEARCHING_PROFESSOR_INITIAL'
    professor_face_box_for_roi = None

    debug_mode = True
    print("\nSystem Initialized. Pipeline: State Machine with Detailed Tracking.")

    mode = 0
    number = -1
    
    while True:
        current_time = time.time()
        fps = cvFpsCalc.get()

        key = cv.waitKey(10) & 0xFF
        if key == 27 or key == ord('q'):
            break
        if key == ord('d'):
            debug_mode = not debug_mode

        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        if current_mode in ['SEARCHING_PROFESSOR_INITIAL', 'REACQUIRING_PROFESSOR']:
            status_text = "INITIAL SEARCH: Looking for Professor..." if current_mode == 'SEARCHING_PROFESSOR_INITIAL' else "RE-ACQUIRING: Looking for Professor..."
            cv.putText(debug_image, status_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            small_frame = cv.resize(image, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    if face_recognition.compare_faces([prof_encoding], face_encoding, tolerance=0.55)[0]:
                        print(f"Professor Found! Switching to TRACKING mode.")
                        top, right, bottom, left = face_location
                        professor_face_box_for_roi = (top * 2, right * 2, bottom * 2, left * 2)
                        current_mode = 'TRACKING_PROFESSOR'
                        break
        
        elif current_mode == 'TRACKING_PROFESSOR':
            f_top, f_right, f_bottom, f_left = professor_face_box_for_roi
            face_width = f_right - f_left
            face_height = f_bottom - f_top
            
            horizontal_padding = int(face_width * 1.5)
            vertical_padding_bottom = int(face_height * 3.0)
            pose_roi_x1 = max(0, f_left - horizontal_padding)
            pose_roi_y1 = max(0, f_top - int(face_height * 0.5))
            pose_roi_x2 = min(cap_width, f_right + horizontal_padding)
            pose_roi_y2 = min(cap_height, f_bottom + vertical_padding_bottom)

            if debug_mode:
                cv.rectangle(debug_image, (pose_roi_x1, pose_roi_y1), (pose_roi_x2, pose_roi_y2), (0, 255, 255), 2)

            pose_input_image = image[pose_roi_y1:pose_roi_y2, pose_roi_x1:pose_roi_x2]
            pose_results = None
            if pose_input_image.size > 0:
                pose_results = pose.process(cv.cvtColor(pose_input_image, cv.COLOR_BGR2RGB))

            if pose_results and pose_results.pose_landmarks:
                cv.putText(debug_image, "TRACKING ACTIVE", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                adjusted_pose_landmarks = copy.deepcopy(pose_results.pose_landmarks)
                for landmark in adjusted_pose_landmarks.landmark:
                    landmark.x = (landmark.x * pose_input_image.shape[1] + pose_roi_x1) / cap_width
                    landmark.y = (landmark.y * pose_input_image.shape[0] + pose_roi_y1) / cap_height
                
                mp_drawing.draw_landmarks(
                    image=debug_image,
                    landmark_list=adjusted_pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                pl = pose_results.pose_landmarks.landmark
                nose, l_eye, r_eye = pl[mp_pose.PoseLandmark.NOSE], pl[mp_pose.PoseLandmark.LEFT_EYE], pl[mp_pose.PoseLandmark.RIGHT_EYE]
                if nose.visibility > 0.6 and l_eye.visibility > 0.6 and r_eye.visibility > 0.6:
                    abs_l_eye_x, abs_r_eye_x, abs_nose_y = (int(l_eye.x * pose_input_image.shape[1]) + pose_roi_x1,
                                                            int(r_eye.x * pose_input_image.shape[1]) + pose_roi_x1,
                                                            int(nose.y * pose_input_image.shape[0]) + pose_roi_y1)
                    eye_dist = abs(abs_l_eye_x - abs_r_eye_x)
                    new_left, new_right = abs_r_eye_x - eye_dist, abs_l_eye_x + eye_dist
                    new_top, new_bottom = abs_nose_y - int(eye_dist * 1.5), abs_nose_y + int(eye_dist * 1.5)
                    professor_face_box_for_roi = (new_top, new_right, new_bottom, new_left)

                landmarks = pose_results.pose_landmarks.landmark    
                if landmarks[mp_pose.PoseLandmark.LEFT_WRIST].visibility > 0.5:
                    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                    wrist_x = int(left_wrist.x * pose_input_image.shape[1]) + pose_roi_x1
                    wrist_y = int(left_wrist.y * pose_input_image.shape[0]) + pose_roi_y1
                    
                    dynamic_hand_roi_box_size = max(150, min(int(face_width * 1.8), 400))
                    hand_roi_half = dynamic_hand_roi_box_size // 2
                    x1, y1 = max(0, wrist_x - hand_roi_half), max(0, wrist_y - hand_roi_half)
                    x2, y2 = min(cap_width, wrist_x + hand_roi_half), min(cap_height, wrist_y + hand_roi_half)
                    if debug_mode:
                        cv.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    hand_region = image[y1:y2, x1:x2]
                    if hand_region.size > 0:
                        hand_results = hands.process(cv.cvtColor(hand_region, cv.COLOR_BGR2RGB))
                        if hand_results.multi_hand_landmarks:
                            hand_landmarks, handedness = hand_results.multi_hand_landmarks[0], hand_results.multi_handedness[0]
                            
                            adj_lm = copy.deepcopy(hand_landmarks)
                            for lm in adj_lm.landmark:
                                lm.x, lm.y = ((lm.x * (x2 - x1) + x1) / cap_width, (lm.y * (y2 - y1) + y1) / cap_height)
                            
                            brect = calc_bounding_rect(debug_image, adj_lm)
                            landmark_list = calc_landmark_list(debug_image, adj_lm)
                            pre_processed_landmark_list = pre_process_landmark(landmark_list)

                            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                            
                            logging_csv(number, mode, pre_processed_landmark_list, [])
                            
                            if hand_sign_id != -1 and (hand_sign_id != last_executed_gesture or (current_time - last_executed_time) > gesture_cooldown):
                                if hand_sign_id == 2:
                                    pyautogui.press("space")
                                    print("[ACTION] Sent 'space'")
                                elif hand_sign_id == 1:
                                    pyautogui.press("f5")
                                    print("[ACTION] Sent 'f5'")
                                elif hand_sign_id == 3 :
                                    pyautogui.press("left")
                                    print("[ACTION] Sent 'left'")
                                last_executed_gesture, last_executed_time = hand_sign_id, current_time
                            
                            if debug_mode:
                                debug_image = draw_landmarks(debug_image, landmark_list)
                                debug_image = draw_info_text(debug_image, brect, handedness, keypoint_classifier_labels[hand_sign_id], "")
                            
                else:
                    pass # Wrist not visible
            else:
                print("Track lost! Attempting to re-acquire professor...")
                current_mode = 'REACQUIRING_PROFESSOR'
                professor_face_box_for_roi = None
                last_executed_gesture = -1

        if debug_mode:
            debug_image = draw_info(debug_image, fps, mode, number)

        cv.imshow('Professor Gesture Control', debug_image)

    cap.release()
    cv.destroyAllWindows()

## Utility Functions

def select_mode(key, mode):
    """
    Selects operation mode based on keyboard input for data logging.
    - 'n': Normal operation (mode 0)
    - 'k': Logging Key Points (mode 1)
    - '0'-'9': Sets the number label for logging in mode 1.
    """
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    if key == ord('n'):
        mode = 0
    if key == ord('k'):
        mode = 1
    return number, mode


def calc_bounding_rect(image, landmarks):
    """Calculates the bounding rectangle around hand landmarks."""
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.zeros((len(landmarks.landmark), 2), dtype=int)

    for i, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array[i] = [landmark_x, landmark_y]

    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    """Converts normalized landmark coordinates to pixel coordinates relative to the image."""
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = [None] * len(landmarks.landmark)

    for i, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point[i] = [landmark_x, landmark_y]

    return landmark_point


def pre_process_landmark(landmark_list):
    """Preprocesses landmark coordinates for the keypoint classifier."""
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    if len(temp_landmark_list) > 0:
        base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]

    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = 0.0
    if len(temp_landmark_list) > 0:
        max_value = max(list(map(abs, temp_landmark_list)))
    
    if max_value == 0:
        return [0.0] * len(temp_landmark_list)

    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list


def logging_csv(number, mode, landmark_list, point_history_list):
    """Logs pre-processed landmark data to CSV files."""
    if mode == 0:
        pass
    elif mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
            print(f"Logged keypoint data for number {number}")
    return


def draw_landmarks(image, landmark_point):
    """Draws hand landmarks and connections on the image."""
    if len(landmark_point) > 0:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5,9), (9,13), (13,17)
        ]

        for connection in connections:
            p1 = landmark_point[connection[0]]
            p2 = landmark_point[connection[1]]
            cv.line(image, tuple(p1), tuple(p2), (0, 0, 0), 6)
            cv.line(image, tuple(p1), tuple(p2), (255, 255, 255), 2)

        for index, landmark in enumerate(landmark_point):
            color_fill = (255, 255, 255)
            color_border = (0, 0, 0)
            radius = 5
            if index in [0, 4, 8, 12, 16, 20]:
                radius = 8
            
            cv.circle(image, (landmark[0], landmark[1]), radius, color_fill, -1)
            cv.circle(image, (landmark[0], landmark[1]), radius, color_border, 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    """Draws the bounding rectangle around the detected hand."""
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text): 
    """Draws information text on the image about the detected hand."""
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


def draw_info(image, fps, mode, number):
    """Draws general information on the image, including FPS, current mode, and logging number."""
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Normal', 'Logging Key Point'] 
    if 0 <= mode <= 1:
        cv.putText(image, "MODE:" + mode_string[mode], (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    
    return image


if __name__ == '__main__':
    main()