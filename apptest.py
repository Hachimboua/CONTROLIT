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
import threading
import queue
import json
import cv2 as cv
import numpy as np
import mediapipe as mp
import face_recognition
import sounddevice as sd
import vosk
# Import Union for type hinting in Python versions < 3.10
from typing import Union 


print("[DIAG] Script execution started.")

try:
    # ... (rest of your imports, ensure the above 'from typing import Union' is there) ...
    print("[DIAG] All required libraries imported.")

    basedir = os.path.dirname(os.path.abspath(sys.argv[0]))
    sys.path.append(os.path.join(basedir, 'model'))
    from model.keypoint_classifier.keypoint_classifier_pyt import KeyPointClassifier
    from utils import CvFpsCalc
    print("[DIAG] Custom gesture model files imported successfully.")

except ImportError as e:
    print(f"\n--- [FATAL IMPORT ERROR] ---", file=sys.stderr)
    print(f"Failed to import a required library: {e}", file=sys.stderr)
    sys.exit(1)


class VoskSpeechController:
    def __init__(self, shutdown_callback, sample_rate: int = 16000):
        print("[DIAG] Initializing VoskSpeechController...")
        self.commands = ["next", "previous", "forward", "backward", "back", "right", "left", "quit", "exit", "close"]
        self.model_path = "vosk-model-small-en-us-0.15"
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.last_detection_time = 0
        self.cooldown_period = 1.5
        self.shutdown_callback = shutdown_callback
        self.keywords = {"next": self._action_next, "previous": self._action_previous, "quit": self._action_quit}
        self.keyword_aliases = {"forward": "next", "right": "next", "backward": "previous", "back": "previous", "left": "previous", "exit": "quit", "close": "quit"}
        self._load_model()
        print("[DIAG] VoskSpeechController Initialized successfully.")

    def _load_model(self):
        print(f"[DIAG] Checking for Vosk model at path: {os.path.abspath(self.model_path)}")
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"Vosk speech model not found at '{self.model_path}'.")
        print("[DIAG] Loading Vosk model into memory...")
        model = vosk.Model(self.model_path)
        grammar = json.dumps(self.commands)
        self.recognizer = vosk.KaldiRecognizer(model, self.sample_rate, grammar)
        self.recognizer.SetWords(True)
        print("[DIAG] Vosk model loaded.")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[WARN] Audio status: {status}", file=sys.stderr)
        self.audio_queue.put(bytes(indata))

    # FIX IS HERE: Changed 'str | None' to 'Union[str, None]'
    def _get_keyword(self, text: str) -> Union[str, None]:
        text = text.lower().strip()
        for k in self.keywords:
            if k in text:
                return k
        for alias, keyword in self.keyword_aliases.items():
            if alias in text:
                return keyword
        return None

    def _execute_action(self, keyword: str):
        current_time = time.time()
        if (current_time - self.last_detection_time) < self.cooldown_period:
            return
        self.last_detection_time = current_time
        print(f"🎤 VOICE COMMAND DETECTED: '{keyword.upper()}'")
        if keyword in self.keywords:
            self.keywords[keyword]()

    def _action_next(self):
        pyautogui.press('right')
        print("[ACTION] Voice: Next slide (→)")

    def _action_previous(self):
        pyautogui.press('left')
        print("[ACTION] Voice: Previous slide (←)")

    def _action_quit(self):
        print("[ACTION] Voice: Quit command received.")
        self.is_running = False
        if self.shutdown_callback:
            self.shutdown_callback()

    def start_listening(self):
        self.is_running = True
        try:
            print("[DIAG] Attempting to open audio stream for speech recognition...")
            with sd.RawInputStream(samplerate=self.sample_rate, channels=1, dtype='int16', callback=self._audio_callback):
                print("\n🔊 SPEECH THREAD: Listening for 'next', 'previous', or 'quit'.\n")
                while self.is_running:
                    data = self.audio_queue.get()
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        if result.get("text"):
                            keyword = self._get_keyword(result["text"])
                            if keyword:
                                self._execute_action(keyword)
        except Exception as e:
            print(f"\n--- [CRITICAL SPEECH THREAD ERROR] ---", file=sys.stderr)
            print(f"ERROR: {e}", file=sys.stderr)
            if self.shutdown_callback:
                self.shutdown_callback()
        finally:
            print("🔇 Speech recognition thread has stopped.")


app_is_running = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()

def main():
    print("[DIAG] main() function started.")
    global app_is_running

    def shutdown_app():
        global app_is_running
        if app_is_running:
            print("[INFO] Shutdown signal received.")
            app_is_running = False

    print("[DIAG] Initializing Speech Controller...")
    speech_controller = VoskSpeechController(shutdown_callback=shutdown_app)
    print("[DIAG] Starting speech thread...")
    speech_thread = threading.Thread(target=speech_controller.start_listening, daemon=True)
    speech_thread.start()

    print("[DIAG] Initializing Gesture Controller...")
    args = get_args()
    print("[DIAG] Opening camera device...")
    cap = cv.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera device {args.device}", file=sys.stderr)
        shutdown_app()
        return
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    print("[DIAG] Camera opened successfully.")

    print("[DIAG] Loading professor.jpg...")
    prof_image = face_recognition.load_image_file("professor.jpg")
    face_encodings = face_recognition.face_encodings(prof_image)
    if not face_encodings:
        raise RuntimeError("No face found in 'professor.jpg'. Ensure the image is clear and contains one face.")
    prof_encoding = face_encodings[0]
    print("[DIAG] professor.jpg loaded successfully.")

    print("[DIAG] Loading MediaPipe models...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=args.use_static_image_mode, max_num_hands=1, min_detection_confidence=args.min_detection_confidence, min_tracking_confidence=args.min_tracking_confidence, model_complexity=0)
    mp_drawing = mp.solutions.drawing_utils
    print("[DIAG] MediaPipe models loaded.")

    print("[DIAG] Loading keypoint classifier...")
    # NOTE: Ensure NUM_CLASSES in keypoint_classifier_pyt.py matches your actual trained model's output classes.
    # If your model predicts 5 classes (0-4), then keypoint_classifier_label.csv must also have 5 entries.
    keypoint_classifier = KeyPointClassifier("model/keypoint_classifier/keypoint_classifier_weights.pth")
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    print("[DIAG] Keypoint classifier loaded.")

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    last_executed_time = 0
    gesture_cooldown = 1.5
    last_executed_gesture = -1
    current_mode = 'SEARCHING_PROFESSOR_INITIAL'
    professor_face_box_for_roi = None
    debug_mode = True
    mode = 0
    number = -1

    print("\n[DIAG] Starting main application loop...")
    while app_is_running:
        current_time = time.time()
        fps = cvFpsCalc.get()
        key = cv.waitKey(10) & 0xFF
        if key == 27 or key == ord('q'):
            shutdown_app()
            break
        if key == ord('d'):
            debug_mode = not debug_mode
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            print("[ERROR] Failed to read from camera.", file=sys.stderr)
            shutdown_app()
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
            if professor_face_box_for_roi is None:
                # Fallback in case ROI box somehow becomes None during tracking
                print("[WARN] Professor face ROI lost, re-entering search mode.")
                current_mode = 'REACQUIRING_PROFESSOR'
                continue # Skip the rest of this iteration

            f_top, f_right, f_bottom, f_left = professor_face_box_for_roi
            face_width = f_right - f_left
            face_height = f_bottom - f_top
            
            horizontal_padding = int(face_width * 1.5)
            vertical_padding_bottom = int(face_height * 3.0)
            pose_roi_x1 = max(0, f_left - horizontal_padding)
            pose_roi_y1 = max(0, f_top - int(face_height * 0.5))
            pose_roi_x2 = min(args.width, f_right + horizontal_padding)
            pose_roi_y2 = min(args.height, f_bottom + vertical_padding_bottom)

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
                    landmark.x = (landmark.x * pose_input_image.shape[1] + pose_roi_x1) / args.width
                    landmark.y = (landmark.y * pose_input_image.shape[0] + pose_roi_y1) / args.height
                
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
                    x2, y2 = min(args.width, wrist_x + hand_roi_half), min(args.height, wrist_y + hand_roi_half)
                    if debug_mode:
                        cv.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    hand_region = image[y1:y2, x1:x2]
                    if hand_region.size > 0:
                        hand_results = hands.process(cv.cvtColor(hand_region, cv.COLOR_BGR2RGB))
                        if hand_results.multi_hand_landmarks:
                            hand_landmarks, handedness = hand_results.multi_hand_landmarks[0], hand_results.multi_handedness[0]
                            
                            adj_lm = copy.deepcopy(hand_landmarks)
                            for lm in adj_lm.landmark:
                                lm.x, lm.y = ((lm.x * (x2 - x1) + x1) / args.width, (lm.y * (y2 - y1) + y1) / args.height)
                            
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
                                # Safely get hand sign text
                                hand_sign_text_to_display = ""
                                if 0 <= hand_sign_id < len(keypoint_classifier_labels):
                                    hand_sign_text_to_display = keypoint_classifier_labels[hand_sign_id]
                                else:
                                    hand_sign_text_to_display = "Unknown Gesture" # Fallback if ID is out of range

                                debug_image = draw_info_text(debug_image, brect, handedness, hand_sign_text_to_display, "")
                            
                else:
                    pass # Wrist not visible
            else:
                print("Track lost! Attempting to re-acquire professor...")
                current_mode = 'REACQUIRING_PROFESSOR'
                professor_face_box_for_roi = None
                last_executed_gesture = -1

        if debug_mode:
            debug_image = draw_info(debug_image, fps, mode, number)

        cv.imshow('CONTROLIT: Professor Gesture & Speech Control', debug_image)

    print("[DIAG] Main loop exited. Cleaning up...")
    cap.release()
    cv.destroyAllWindows()
    if speech_thread and speech_thread.is_alive():
        # Give the speech thread a moment to finish gracefully
        speech_controller.is_running = False # Signal it to stop
        speech_thread.join(timeout=2.0) # Wait for it to join, with a timeout
        if speech_thread.is_alive():
            print("[WARN] Speech thread did not terminate gracefully.", file=sys.stderr)
    print("[DIAG] Application shut down cleanly.")

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
    landmark_array = np.empty((len(landmarks.landmark), 2), dtype=int)
    for i, landmark in enumerate(landmarks.landmark):
        landmark_array[i] = [min(int(landmark.x * image_width), image_width - 1), min(int(landmark.y * image_height), image_height - 1)]
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[min(int(lm.x * image_width), image_width - 1), min(int(lm.y * image_height), image_height - 1)] for lm in landmarks.landmark]

def pre_process_landmark(landmark_list):
    temp_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_list[0][0], temp_list[0][1]
    for i in range(len(temp_list)):
        temp_list[i][0] -= base_x
        temp_list[i][1] -= base_y
    temp_list_flat = list(itertools.chain.from_iterable(temp_list))
    max_val = max(map(abs, temp_list_flat)) if temp_list_flat else 1
    return [n / max_val for n in temp_list_flat]

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 1 and (0 <= number <= 9):
        with open('model/keypoint_classifier/keypoint.csv', 'a', newline="") as f:
            csv.writer(f).writerow([number, *landmark_list])

def draw_landmarks(image, landmark_point):
    """Draws hand landmarks and connections on the image."""
    if len(landmark_point) > 0:
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17)            # Palm base connections
        ]

        # Draw connections
        for connection in connections:
            p1 = landmark_point[connection[0]]
            p2 = landmark_point[connection[1]]
            cv.line(image, tuple(p1), tuple(p2), (0, 0, 0), 6)  # Black thicker line
            cv.line(image, tuple(p1), tuple(p2), (255, 255, 255), 2) # White thinner line

        # Draw landmarks
        for index, landmark in enumerate(landmark_point):
            color_fill = (255, 255, 255) # White fill
            color_border = (0, 0, 0)     # Black border
            radius = 5
            # Make certain keypoints larger (e.g., fingertips, wrist base)
            if index in [0, 4, 8, 12, 16, 20]:
                radius = 8
            
            cv.circle(image, (landmark[0], landmark[1]), radius, color_fill, -1) # Filled circle
            cv.circle(image, (landmark[0], landmark[1]), radius, color_border, 1) # Border

    return image

def draw_bounding_rect(use_brect, image, brect):
    """Draws the bounding rectangle around the detected hand."""
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text): 
    """Draws information text on the image about the detected hand."""
    # Background rectangle for the text
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    # Handedness label (e.g., "Left", "Right")
    info_text = handedness.classification[0].label[0:]
    # Add hand sign text if available
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    
    # Put the text on the image
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

def draw_info(image, fps, mode, number):
    """Draws general information on the image, including FPS, current mode, and logging number."""
    # FPS text
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA) # Black outline
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA) # White fill

    # Mode text
    mode_string = ['Normal', 'Logging Key Point'] 
    if 0 <= mode <= 1:
        cv.putText(image, "MODE:" + mode_string[mode], (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        # Logging number text (only in logging mode)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    
    return image


print("\n[DIAG] Reached end of script file. Forcing main() function to run now...")

try:
    main()
except Exception as e:
    print(f"\n--- [FATAL ERROR IN MAIN EXECUTION] ---", file=sys.stderr)
    print(f"An unexpected error occurred while running the main() function: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)