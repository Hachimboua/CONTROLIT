#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import itertools
import pyautogui
import time
import os
import torch
import sys

import cv2 as cv
import numpy as np
import mediapipe as mp
import face_recognition

# Add the model directory to path to ensure classifiers can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

# Import your model classes (ensure these paths are correct relative to model/ directory)
from model.keypoint_classifier.keypoint_classifier_pyt import KeyPointClassifier
from utils import CvFpsCalc # Assuming CvFpsCalc is in a utils.py in a utils/ directory

class VisionBackend:
    """
    Encapsulates the entire computer vision pipeline for gesture and face recognition.
    This class is designed to run in a separate thread and communicate with a
    GUI via a queue.
    """
    def __init__(self, cap_device=0, cap_width=960, cap_height=540):
        # --- Store arguments and configuration ---
        self.cap_device = cap_device
        self.cap_width = cap_width
        self.cap_height = cap_height
        self.use_static_image_mode = False
        self.min_hand_detection_confidence = 0.7
        self.min_hand_tracking_confidence = 0.5
        self.use_brect = True # From original script

        # --- Camera, Models, and Classifiers Setup ---
        self.cap = cv.VideoCapture(self.cap_device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)

        professor_image_path = "professor.jpg"
        try:
            prof_image = face_recognition.load_image_file(professor_image_path)
            prof_face_locations = face_recognition.face_locations(prof_image, model="hog")
            self.prof_encoding = face_recognition.face_encodings(prof_image, prof_face_locations, num_jitters=50)[0]
        except Exception as e:
            raise IOError(f"FATAL: Could not load professor image from {professor_image_path}. Error: {e}")

        # --- MediaPipe Models ---
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.use_static_image_mode, max_num_hands=1,
            min_detection_confidence=self.min_hand_detection_confidence, 
            min_tracking_confidence=self.min_hand_tracking_confidence,
            model_complexity=0
        )

        # --- Gesture Classifier ---
        try:
            self.keypoint_classifier = KeyPointClassifier("model/keypoint_classifier/keypoint_classifier_weights.pth")
            with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
                self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
        except Exception as e:
            raise IOError(f"FATAL: Could not load gesture models. Error: {e}")

        # --- FPS Counter ---
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)

    def run(self, gui_queue, stop_event):
        """
        Main processing loop. Sends data to the GUI via the queue.
        - 'gui_queue' is a queue.Queue() object for thread-safe communication.
        - 'stop_event' is a threading.Event() object to signal when to stop.
        """
        # --- State Machine & Timing Variables ---
        last_executed_time = 0
        gesture_cooldown = 1.5
        last_executed_gesture = -1
        current_mode = 'SEARCHING_PROFESSOR_INITIAL'
        professor_face_box_for_roi = None
        debug_mode = True 

        gui_queue.put(("status", "System Initialized. Press 'Start' in GUI."))

        while not stop_event.is_set():
            current_time = time.time()
            fps = self.cvFpsCalc.get()

            ret, image = self.cap.read()
            if not ret:
                gui_queue.put(("status", "Error: Camera feed lost. Stopping."))
                break
            
            image = cv.flip(image, 1)
            debug_image = copy.deepcopy(image)

            # ##################################################################
            # ## STATE: SEARCHING_PROFESSOR_INITIAL or REACQUIRING_PROFESSOR ##
            # ##################################################################
            if current_mode in ['SEARCHING_PROFESSOR_INITIAL', 'REACQUIRING_PROFESSOR']:
                status_text = "INITIAL SEARCH: Looking for Professor..." if current_mode == 'SEARCHING_PROFESSOR_INITIAL' else "RE-ACQUIRING: Looking for Professor..."
                
                small_frame = cv.resize(image, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                
                found_prof = False
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    for face_encoding, face_location in zip(face_encodings, face_locations):
                        if face_recognition.compare_faces([self.prof_encoding], face_encoding, tolerance=0.55)[0]:
                            gui_queue.put(("status", "Professor Found! Switching to TRACKING mode."))
                            top, right, bottom, left = face_location
                            professor_face_box_for_roi = (top * 2, right * 2, bottom * 2, left * 2)
                            current_mode = 'TRACKING_PROFESSOR'
                            found_prof = True
                            break
                if not found_prof:
                     gui_queue.put(("status", status_text))

            # ###########################################
            # ## STATE: TRACKING_PROFESSOR             ##
            # ###########################################
            elif current_mode == 'TRACKING_PROFESSOR':
                f_top, f_right, f_bottom, f_left = professor_face_box_for_roi
                face_width = f_right - f_left
                face_height = f_bottom - f_top
                
                horizontal_padding = int(face_width * 1.5)
                vertical_padding_bottom = int(face_height * 3.0)
                pose_roi_x1 = max(0, f_left - horizontal_padding)
                pose_roi_y1 = max(0, f_top - int(face_height * 0.5))
                pose_roi_x2 = min(self.cap_width, f_right + horizontal_padding)
                pose_roi_y2 = min(self.cap_height, f_bottom + vertical_padding_bottom)

                if debug_mode:
                    cv.rectangle(debug_image, (pose_roi_x1, pose_roi_y1), (pose_roi_x2, pose_roi_y2), (0, 255, 255), 2)

                pose_input_image = image[pose_roi_y1:pose_roi_y2, pose_roi_x1:pose_roi_x2]
                pose_results = None
                if pose_input_image.size > 0:
                    pose_results = self.pose.process(cv.cvtColor(pose_input_image, cv.COLOR_BGR2RGB))

                if pose_results and pose_results.pose_landmarks:
                    gui_queue.put(("status", f"TRACKING ACTIVE (FPS: {int(fps)})"))
                    
                    adjusted_pose_landmarks = copy.deepcopy(pose_results.pose_landmarks)
                    for landmark in adjusted_pose_landmarks.landmark:
                        landmark.x = (landmark.x * pose_input_image.shape[1] + pose_roi_x1) / self.cap_width
                        landmark.y = (landmark.y * pose_input_image.shape[0] + pose_roi_y1) / self.cap_height
                    
                    self.mp_drawing.draw_landmarks(
                        image=debug_image, landmark_list=adjusted_pose_landmarks,
                        connections=self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    )

                    # --- Self-Correcting ROI Update from Pose ---
                    # THIS IS THE CORRECTED DYNAMIC ROI LOGIC
                    pl = pose_results.pose_landmarks.landmark
                    nose = pl[self.mp_pose.PoseLandmark.NOSE]
                    l_eye = pl[self.mp_pose.PoseLandmark.LEFT_EYE]
                    r_eye = pl[self.mp_pose.PoseLandmark.RIGHT_EYE]
                    
                    if nose.visibility > 0.6 and l_eye.visibility > 0.6 and r_eye.visibility > 0.6:
                        abs_l_eye_x = int(l_eye.x * pose_input_image.shape[1]) + pose_roi_x1
                        abs_r_eye_x = int(r_eye.x * pose_input_image.shape[1]) + pose_roi_x1
                        abs_nose_y = int(nose.y * pose_input_image.shape[0]) + pose_roi_y1
                        
                        eye_dist = abs(abs_l_eye_x - abs_r_eye_x)
                        
                        new_left = abs_r_eye_x - eye_dist
                        new_right = abs_l_eye_x + eye_dist
                        new_top = abs_nose_y - int(eye_dist * 1.5)
                        new_bottom = abs_nose_y + int(eye_dist * 1.5)
                        
                        # Update the main ROI box for the next frame
                        professor_face_box_for_roi = (new_top, new_right, new_bottom, new_left)

                    # --- Strict Left-Hand-Only Gesture Processing ---
                    landmarks = pose_results.pose_landmarks.landmark
                    if landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].visibility > 0.5:
                        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                        wrist_x = int(left_wrist.x * pose_input_image.shape[1]) + pose_roi_x1
                        wrist_y = int(left_wrist.y * pose_input_image.shape[0]) + pose_roi_y1
                        
                        dynamic_hand_roi_box_size = max(150, min(int(face_width * 1.8), 400))
                        hand_roi_half = dynamic_hand_roi_box_size // 2
                        x1, y1 = max(0, wrist_x - hand_roi_half), max(0, wrist_y - hand_roi_half)
                        x2, y2 = min(self.cap_width, wrist_x + hand_roi_half), min(self.cap_height, wrist_y + hand_roi_half)
                        if debug_mode: cv.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        hand_region = image[y1:y2, x1:x2]
                        if hand_region.size > 0:
                            hand_results = self.hands.process(cv.cvtColor(hand_region, cv.COLOR_BGR2RGB))
                            if hand_results.multi_hand_landmarks:
                                hand_landmarks, handedness = hand_results.multi_hand_landmarks[0], hand_results.multi_handedness[0]
                                adj_lm = copy.deepcopy(hand_landmarks)
                                for lm in adj_lm.landmark:
                                    lm.x, lm.y = ((lm.x * (x2 - x1) + x1) / self.cap_width, (lm.y * (y2 - y1) + y1) / self.cap_height)
                                
                                brect = calc_bounding_rect(debug_image, adj_lm)
                                landmark_list = calc_landmark_list(debug_image, adj_lm)
                                hand_sign_id = self.keypoint_classifier(pre_process_landmark(landmark_list))
                                
                                if hand_sign_id != -1 and (hand_sign_id != last_executed_gesture or (current_time - last_executed_time) > gesture_cooldown):
                                    action_text = ""
                                    if hand_sign_id == 3:
                                        pyautogui.press("space")
                                        action_text = "Sent 'space'"
                                    elif hand_sign_id == 1:
                                        pyautogui.press("f5")
                                        action_text = "Sent 'f5'"
                                    
                                    if action_text:
                                        gui_queue.put(("action", f"[ACTION] {action_text}"))
                                        last_executed_gesture, last_executed_time = hand_sign_id, current_time
                                
                                if debug_mode:
                                    debug_image = draw_landmarks(debug_image, landmark_list)
                                    debug_image = draw_info_text(debug_image, brect, handedness, self.keypoint_classifier_labels[hand_sign_id], "")
                else:
                    gui_queue.put(("status", "Track lost! Attempting to re-acquire professor..."))
                    current_mode = 'REACQUIRING_PROFESSOR'
                    professor_face_box_for_roi = None
                    last_executed_gesture = -1

            gui_queue.put(("image", cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)))

        self.cap.release()
        gui_queue.put(("status", "Vision thread stopped."))


# ####################################################################
# ## UTILITY FUNCTIONS (Copied directly from the original script)  ##
# ####################################################################

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    if key == ord('n'):
        mode = 0
    if key == ord('k'):
        mode = 1
    if key == ord('h'):
        mode = 2
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((len(landmarks.landmark), 2), dtype=int)
    for i, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array[i] = [landmark_x, landmark_y]
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    if temp_landmark_list:
        base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list))) if temp_landmark_list else 1
    def normalize(n):
        return n / max_value
    temp_landmark_list = list(map(normalize, temp_landmark_list))
    return temp_landmark_list

def draw_landmarks(image, landmark_point):
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
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), 
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image
