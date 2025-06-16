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
import json
from typing import Optional

import cv2 as cv
import numpy as np
import mediapipe as mp
import face_recognition

import sounddevice as sd
import vosk

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from model.keypoint_classifier.keypoint_classifier_pyt import KeyPointClassifier
from utils import CvFpsCalc

# --- Global State Variables ---
app_is_running = True
is_speech_mode_active = False
last_speech_gesture_time = 0
last_speech_action_time = 0

# --- Speech Recognition Helper Functions ---
def get_keyword(text: str) -> Optional[str]:
    words = text.lower().strip().split()
    keywords = {"next": "next", "previous": "previous", "quit": "quit"}
    aliases = {"forward": "next", "right": "next", "backward": "previous", "back": "previous", "left": "previous", "exit": "quit", "close": "quit"}
    
    for k in keywords:
        if k in words:
            return k
    for alias, keyword in aliases.items():
        if alias in words:
            return keyword
    return None

def execute_speech_action(recognized_text: str):
    global last_speech_action_time, last_speech_gesture_time
    
    if not is_speech_mode_active:
        return

    keyword = get_keyword(recognized_text)
    if not keyword:
        return

    current_time = time.time()
    if (current_time - last_speech_action_time) < 1.5:
        return
    
    last_speech_action_time = current_time
    print(f"🎤 VOICE COMMAND: '{keyword.upper()}' (from text: '{recognized_text}')")
    
    if keyword == 'next':
        pyautogui.press('right')
        print("[ACTION] Voice: Next slide (→)")
    elif keyword == 'previous':
        pyautogui.press('left')
        print("[ACTION] Voice: Previous slide (←)")
    elif keyword == 'quit':
        shutdown_app()
    
    last_speech_gesture_time = current_time

def shutdown_app():
    global app_is_running
    if app_is_running:
        print("[INFO] Shutdown signal received. Exiting...")
        app_is_running = False

# --- Your Original Utility Functions ---
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--width", help='Camera capture width', type=int, default=960)
    parser.add_argument("--height", help='Camera capture height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    if key == ord('n'):
        mode = 0
    if key == ord('k'):
        mode = 1
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.zeros((len(landmarks.landmark), 2), dtype=int)
    for i, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array[i] = [landmark_x, landmark_y]
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = [None] * len(landmarks.landmark)
    for i, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point[i] = [landmark_x, landmark_y]
    return landmark_point

def pre_process_landmark(landmark_list):
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
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
            print(f"Logged keypoint data for number {number}")
    return

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5,9), (9,13), (13,17)]
        for connection in connections:
            p1 = landmark_point[connection[0]]
            p2 = landmark_point[connection[1]]
            cv.line(image, tuple(p1), tuple(p2), (0, 0, 0), 6)
            cv.line(image, tuple(p1), tuple(p2), (255, 255, 255), 2)
        for index, landmark in enumerate(landmark_point):
            radius = 8 if index in [0, 4, 8, 12, 16, 20] else 5
            cv.circle(image, (landmark[0], landmark[1]), radius, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    mode_string = ['Normal', 'Logging Key Point']
    if 0 <= mode <= 1:
        cv.putText(image, "MODE:" + mode_string[mode], (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

def main():
    global app_is_running, is_speech_mode_active, last_speech_gesture_time, last_speech_action_time

    # --- INITIALIZATION ---
    try:
        print("[INFO] Initializing Vosk Model...")
        model_path = "vosk-model-small-en-us-0.15"
        if not os.path.exists(model_path): raise RuntimeError(f"Vosk model not found at '{model_path}'.")
        model = vosk.Model(model_path)
        recognizer = vosk.KaldiRecognizer(model, 16000)
        print("[INFO] Vosk Model Initialized.")
    except Exception as e:
        print(f"FATAL: Could not start Vosk: {e}"); return

    args = get_args()
    # FIX: Restore the definitions for cap_width and cap_height right after parsing args
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

    try:
        professor_image_path = "professor.jpg"
        prof_image = face_recognition.load_image_file(professor_image_path)
        prof_encoding = face_recognition.face_encodings(prof_image)[0]
    except Exception as e:
        print(f"FATAL: Could not load professor image: {e}"); return

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=min_hand_detection_confidence, min_tracking_confidence=min_hand_tracking_confidence)
    
    try:
        keypoint_classifier = KeyPointClassifier("model/keypoint_classifier/keypoint_classifier_weights.pth")
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    except Exception as e:
        print(f"FATAL: Could not load gesture models: {e}"); return

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    last_executed_time = 0
    gesture_cooldown = 1.5
    last_executed_gesture = -1
    current_mode = 'SEARCHING_PROFESSOR_INITIAL'
    professor_face_box_for_roi = None
    debug_mode = True
    mode = 0
    number = -1
    
    try:
        with sd.RawInputStream(samplerate=16000, blocksize=4000, device=None, dtype='int16', channels=1) as stream:
            print("\nSystem Initialized. All systems running in a single thread.")
            
            while app_is_running:
                current_time = time.time()
                fps = cvFpsCalc.get()
                key = cv.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'): shutdown_app()
                if key == ord('d'): debug_mode = not debug_mode
                number, mode = select_mode(key, mode)

                if is_speech_mode_active and (current_time - last_speech_gesture_time > 5.0):
                    print("[INFO] Speech mode timed out.")
                    is_speech_mode_active = False

                ret, image = cap.read()
                if not ret: shutdown_app()
                if not app_is_running: break

                image = cv.flip(image, 1)
                debug_image = copy.deepcopy(image)

                # --- SINGLE-THREADED SPEECH RECOGNITION ---
                if is_speech_mode_active:
                    # Read a chunk of audio data while the video frame is being processed
                    audio_data, overflowed = stream.read(stream.blocksize)
                    if overflowed: print("[WARN] Audio overflow!")
                    if recognizer.AcceptWaveform(bytes(audio_data)):
                        result = json.loads(recognizer.Result())
                        if result.get("text"):
                            print(f"[SPEECH RAW] Vosk heard: \"{result['text']}\"")
                            execute_speech_action(result['text'])
                
                # --- YOUR FULL GESTURE LOGIC IS PRESERVED HERE ---
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
                        mp_drawing.draw_landmarks(image=debug_image, landmark_list=adjusted_pose_landmarks, connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
                        pl = pose_results.pose_landmarks.landmark
                        nose, l_eye, r_eye = pl[mp_pose.PoseLandmark.NOSE], pl[mp_pose.PoseLandmark.LEFT_EYE], pl[mp_pose.PoseLandmark.RIGHT_EYE]
                        if nose.visibility > 0.6 and l_eye.visibility > 0.6 and r_eye.visibility > 0.6:
                            abs_l_eye_x, abs_r_eye_x, abs_nose_y = (int(l_eye.x * pose_input_image.shape[1]) + pose_roi_x1, int(r_eye.x * pose_input_image.shape[1]) + pose_roi_x1, int(nose.y * pose_input_image.shape[0]) + pose_roi_y1)
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
                                    
                                    if hand_sign_id != -1 and (current_time - last_executed_time) > gesture_cooldown:
                                        if hand_sign_id == 0:
                                            print("[ACTION] Gesture: Speech Mode Activated!")
                                            is_speech_mode_active = True
                                            last_speech_gesture_time = current_time
                                        elif hand_sign_id == 2:
                                            pyautogui.press("space")
                                            print("[ACTION] Sent 'space'")
                                        elif hand_sign_id == 1:
                                            pyautogui.press("f5")
                                            print("[ACTION] Sent 'f5'")
                                        elif hand_sign_id == 3:
                                            pyautogui.press("left")
                                            print("[ACTION] Sent 'left'")
                                        last_executed_time = current_time
                                        last_executed_gesture = hand_sign_id
                                    
                                    if debug_mode:
                                        debug_image = draw_landmarks(debug_image, landmark_list)
                                        debug_image = draw_info_text(debug_image, brect, handedness, keypoint_classifier_labels[hand_sign_id], "")
                    else:
                        print("Track lost! Attempting to re-acquire professor...")
                        current_mode = 'REACQUIRING_PROFESSOR'
                        professor_face_box_for_roi = None
                        last_executed_gesture = -1

                # --- Drawing and Display Logic ---
                if debug_mode:
                    draw_info(debug_image, fps, mode, number)
                    speech_status = "Speech: ACTIVE" if is_speech_mode_active else "Speech: INACTIVE"
                    color = (0, 255, 0) if is_speech_mode_active else (0, 0, 255)
                    cv.putText(debug_image, speech_status, (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
                    cv.putText(debug_image, speech_status, (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                cv.imshow('Professor Gesture & Speech Control', debug_image)

    except Exception as e:
        print(f"A fatal error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv.destroyAllWindows()
        print("Application Closed.")

# --- SCRIPT ENTRY POINT ---
if __name__ == '__main__':
    main()