import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import json
import math
from collections import deque
# from scipy.spatial.distance import cosine # Not used directly, numpy dot product used
import time
import traceback
import os
# Removed argparse, replaced by tkinter GUI
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# --- Configuration (Keep these easily accessible) ---
MODEL_PATH = 'yolov8n.pt'
VIDEO_SOURCE = 0 # Webcam default, change as needed (e.g., 'path/to/video.mp4')
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
MP_POSE_MODEL_COMPLEXITY = 1 # 0, 1, 2 (Higher = more accurate, slower)
MP_POSE_MIN_DETECTION_CONFIDENCE = 0.5
MP_POSE_MIN_TRACKING_CONFIDENCE = 0.5
FEATURE_BUFFER_SIZE = 60 # Frames to buffer for identity vector generation
IDENTITY_VECTOR_DIM = 54 # 18 means + 18 stds + 18 medians
MATCH_THRESHOLD = 0.70 # Cosine similarity threshold for mapping match
DB_FILE = 'employee_enhanced_db.json'
DRAW_POSE_LANDMARKS = True # Toggle drawing pose landmarks in mapping mode

# --- Visualization Settings ---
# General
TEXT_COLOR = (255, 255, 255) # White
POSE_COLOR = (0, 255, 0) # Green for landmarks
POSE_CONNECTIONS_COLOR = (0, 0, 255) # Blue for connections
# Registration Mode
REG_BBOX_COLOR = (0, 255, 255) # Yellow
REG_TEXT_COLOR = (0, 0, 0) # Black text on instructions box
REG_INFO_BOX_COLOR = (0, 255, 255) # Yellow background for instructions
# Mapping Mode
MAP_BBOX_COLOR_UNKNOWN = (0, 255, 0) # Green (Default/Unknown/Ready to Match)
MAP_BBOX_COLOR_KNOWN = (255, 0, 255) # Magenta (Identified)
MAP_BBOX_COLOR_PROCESSING = (255, 255, 0) # Cyan (Buffer filling)


# --- MediaPipe and Helper Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
LM = mp_pose.PoseLandmark

# Define which landmarks to use for angles and proportions
ANGLE_INDICES = [
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST), (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    (LM.LEFT_ELBOW, LM.LEFT_SHOULDER, LM.LEFT_HIP), (LM.RIGHT_ELBOW, LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE), (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    (LM.LEFT_SHOULDER, LM.LEFT_HIP, LM.LEFT_KNEE), (LM.RIGHT_SHOULDER, LM.RIGHT_HIP, LM.RIGHT_KNEE),
    # Optional: Add more complex angles if needed
    (LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER) # Head/Shoulder angle
]

PROPORTION_INDICES = [
    # Arm length relative to torso height (shoulder to hip)
    ((LM.LEFT_SHOULDER, LM.LEFT_WRIST), (LM.LEFT_SHOULDER, LM.LEFT_HIP)),
    ((LM.RIGHT_SHOULDER, LM.RIGHT_WRIST), (LM.RIGHT_SHOULDER, LM.RIGHT_HIP)),
    # Leg length relative to torso height
    ((LM.LEFT_HIP, LM.LEFT_ANKLE), (LM.LEFT_SHOULDER, LM.LEFT_HIP)),
    ((LM.RIGHT_HIP, LM.RIGHT_ANKLE), (LM.RIGHT_SHOULDER, LM.RIGHT_HIP)),
    # Shoulder width relative to hip width
    ((LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER), (LM.LEFT_HIP, LM.RIGHT_HIP)),
    # Arm length relative to leg length
    ((LM.LEFT_SHOULDER, LM.LEFT_WRIST), (LM.LEFT_HIP, LM.LEFT_ANKLE)),
    ((LM.RIGHT_SHOULDER, LM.RIGHT_WRIST), (LM.RIGHT_HIP, LM.RIGHT_ANKLE)),
    # Approximate height (nose to avg ankle) relative to shoulder width
    ((LM.NOSE, (LM.LEFT_ANKLE, LM.RIGHT_ANKLE)), (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)),
    # Torso "height" (shoulder midpoint to hip midpoint) relative to hip width
    (((LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER), (LM.LEFT_HIP, LM.RIGHT_HIP)), (LM.LEFT_HIP, LM.RIGHT_HIP))
]

# Ensure the total number of features matches IDENTITY_VECTOR_DIM / 3
assert len(ANGLE_INDICES) + len(PROPORTION_INDICES) == IDENTITY_VECTOR_DIM // 3, \
    f"Feature definition mismatch: {len(ANGLE_INDICES)} angles + {len(PROPORTION_INDICES)} proportions != {IDENTITY_VECTOR_DIM // 3}"

# --- Helper Functions ---

def calculate_angle(p1, p2, p3):
    """Calculates the angle between three 2D points."""
    if p1 is None or p2 is None or p3 is None: return None
    try:
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return None
        dot_product = np.dot(v1, v2)
        cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return angle if not (np.isnan(angle) or np.isinf(angle)) else None
    except Exception:
        # traceback.print_exc() # Uncomment for debugging angle calculation errors
        return None

def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two points (Landmark objects or dicts)."""
    if p1 is None or p2 is None: return None
    try:
        # Handle both Landmark objects and dicts with 'x', 'y' keys
        x1 = p1.x if hasattr(p1, 'x') else p1['x']
        y1 = p1.y if hasattr(p1, 'y') else p1['y']
        x2 = p2.x if hasattr(p2, 'x') else p2['x']
        y2 = p2.y if hasattr(p2, 'y') else p2['y']

        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return dist if not (np.isnan(dist) or np.isinf(dist)) else None
    except (AttributeError, KeyError, Exception):
        # traceback.print_exc() # Uncomment for debugging distance calculation errors
        return None

def get_landmark(landmarks, index):
    """Safely retrieves a landmark if visibility is sufficient."""
    if not isinstance(index, LM): return None # Ensure index is a PoseLandmark enum
    if landmarks and 0 <= index.value < len(landmarks.landmark):
        lm = landmarks.landmark[index.value]
        # Use a slightly lower threshold here than tracking confidence for feature extraction
        if lm.visibility > (MP_POSE_MIN_TRACKING_CONFIDENCE * 0.8):
            return lm
    return None

def get_average_landmark(landmarks, index1, index2):
    """Calculates the average position of two landmarks."""
    p1 = get_landmark(landmarks, index1)
    p2 = get_landmark(landmarks, index2)
    if p1 and p2:
        # Create a simple object mimicking Landmark structure
        return type('obj', (object,), {
            'x': (p1.x + p2.x) / 2, 'y': (p1.y + p2.y) / 2,
            'z': (p1.z + p2.z) / 2, 'visibility': (p1.visibility + p2.visibility) / 2
        })
    # Fallback: return one if the other is missing? Or None? Returning None is safer.
    # elif p1: return p1
    # elif p2: return p2
    return None

def get_point_from_definition(landmarks, definition):
    """Gets a point, which can be a single landmark or the average of two."""
    if isinstance(definition, LM):
        return get_landmark(landmarks, definition)
    elif isinstance(definition, tuple) and len(definition) == 2 and \
         isinstance(definition[0], LM) and isinstance(definition[1], LM):
        return get_average_landmark(landmarks, definition[0], definition[1])
    else:
        # print(f"Warning: Invalid point definition: {definition}") # Debugging
        return None

def calculate_proportion(landmarks, p_indices):
    """Calculates the ratio of distances between two pairs of points."""
    try:
        # Validate input structure
        if not (isinstance(p_indices, tuple) and len(p_indices) == 2): return None
        dist1_def, dist2_def = p_indices
        if not (isinstance(dist1_def, tuple) and len(dist1_def) == 2 and
                isinstance(dist2_def, tuple) and len(dist2_def) == 2): return None

        # Get points for the first distance
        p1a = get_point_from_definition(landmarks, dist1_def[0])
        p1b = get_point_from_definition(landmarks, dist1_def[1])
        dist1 = calculate_distance(p1a, p1b)

        # Get points for the second distance
        p2a = get_point_from_definition(landmarks, dist2_def[0])
        p2b = get_point_from_definition(landmarks, dist2_def[1])
        dist2 = calculate_distance(p2a, p2b)

        # Calculate ratio safely
        if dist1 is not None and dist2 is not None and dist2 > 1e-6: # Avoid division by zero
            ratio = dist1 / dist2
            # Check for invalid float results
            return ratio if not (np.isnan(ratio) or np.isinf(ratio)) else None
        return None
    except Exception:
        # traceback.print_exc() # Uncomment for debugging proportion calculation
        return None

def extract_pose_features(landmarks):
    """Extracts angles and proportions from pose landmarks."""
    if landmarks is None:
        # Return zeros if no landmarks detected for this frame
        return np.zeros(len(ANGLE_INDICES) + len(PROPORTION_INDICES), dtype=np.float32)

    features = []
    # Calculate Angles
    for angle_def in ANGLE_INDICES:
        p1 = get_point_from_definition(landmarks, angle_def[0])
        p2 = get_point_from_definition(landmarks, angle_def[1])
        p3 = get_point_from_definition(landmarks, angle_def[2])
        angle = calculate_angle(p1, p2, p3)
        features.append(angle if angle is not None else 0.0) # Use 0 for failed calculations

    # Calculate Proportions
    for prop_indices in PROPORTION_INDICES:
        proportion = calculate_proportion(landmarks, prop_indices)
        features.append(proportion if proportion is not None else 0.0) # Use 0 for failed calculations

    # Normalize features (example: angles 0-1, proportions capped)
    normalized_features = []
    angle_count = len(ANGLE_INDICES)
    for i, feat in enumerate(features):
        if i < angle_count:
            # Normalize angles to [0, 1]
            normalized_features.append(np.clip(feat / 180.0, 0.0, 1.0))
        else:
            # Normalize proportions - clip to a reasonable range (e.g., 0 to 5)
            # This range might need tuning based on observed values
            normalized_features.append(np.clip(feat, 0.0, 5.0))

    return np.array(normalized_features, dtype=np.float32)

def generate_identity_vector(feature_buffer):
    """Generates a stable identity vector from a buffer of features."""
    required_len = FEATURE_BUFFER_SIZE // 2 # Require at least half the buffer filled
    if not feature_buffer or len(feature_buffer) < required_len:
        # print(f"Debug: Buffer length {len(feature_buffer)} insufficient (requires {required_len}).")
        return None

    # Filter out potential invalid entries (e.g., None or incorrect shape)
    feature_dim = IDENTITY_VECTOR_DIM // 3
    valid_rows = [row for row in feature_buffer if isinstance(row, np.ndarray) and row.shape == (feature_dim,)]

    if len(valid_rows) < required_len:
        # print(f"Debug: Valid rows {len(valid_rows)} insufficient (requires {required_len}).")
        return None

    buffer_np = np.array(valid_rows, dtype=np.float32)

    try:
        # Calculate mean, std deviation, and median for each feature dimension
        # Use nan-aware functions if buffer might contain NaNs (though extract_pose_features aims to avoid them)
        means = np.nanmean(buffer_np, axis=0)
        stds = np.nanstd(buffer_np, axis=0)
        medians = np.nanmedian(buffer_np, axis=0)

        # Handle potential NaNs resulting from calculations (e.g., if a column was all NaN)
        means = np.nan_to_num(means)
        stds = np.nan_to_num(stds)
        medians = np.nan_to_num(medians)

        # Concatenate statistics to form the identity vector
        identity_vector = np.concatenate([means, stds, medians]).astype(np.float32)

        # Final validation of shape
        if len(identity_vector) != IDENTITY_VECTOR_DIM:
             print(f"Error: Generated identity vector has wrong dimension: {len(identity_vector)}, expected {IDENTITY_VECTOR_DIM}")
             return None

        # Normalize the final identity vector (L2 norm) for cosine similarity comparison
        norm = np.linalg.norm(identity_vector)
        if norm < 1e-6: # Avoid division by zero if vector is all zeros
             return np.zeros_like(identity_vector)

        id_vec_norm = identity_vector / norm

        # Final check for NaN/Inf in the normalized vector
        if np.any(np.isnan(id_vec_norm)) or np.any(np.isinf(id_vec_norm)):
            print("Warning: NaN or Inf detected in final normalized identity vector. Returning None.")
            return None

        return id_vec_norm

    except Exception as e:
        print(f"Error generating identity vector: {e}")
        traceback.print_exc()
        return None

def match_identity(live_vector, employee_db):
    """Matches a live identity vector against the database using cosine similarity."""
    best_match_id = None
    best_match_name = "Unknown"
    highest_similarity = -1.0 # Initialize below possible cosine range

    # Validate live vector
    if live_vector is None or len(live_vector) != IDENTITY_VECTOR_DIM:
        # print("Debug: Invalid live vector for matching.")
        return best_match_id, best_match_name, highest_similarity
    # Ensure live vector is not zero vector (already normalized, but double check norm)
    # norm_live = np.linalg.norm(live_vector) # Live vector should already be normalized
    # if norm_live < 1e-6:
    #     print("Debug: Live vector is zero vector.")
    #     return best_match_id, best_match_name, highest_similarity

    for emp_id, data in employee_db.items():
        stored_vector_list = data.get('enhanced_feature')
        if not isinstance(stored_vector_list, list) or len(stored_vector_list) != IDENTITY_VECTOR_DIM:
            # print(f"Warning: Skipping Emp ID {emp_id} due to invalid stored vector format.")
            continue # Skip invalid entries

        stored_vector = np.array(stored_vector_list, dtype=np.float32)

        # Stored vector should ideally be pre-normalized, but normalize again just in case
        norm_stored = np.linalg.norm(stored_vector)
        if norm_stored < 1e-6:
             # print(f"Warning: Skipping Emp ID {emp_id} due to zero vector in DB.")
             continue # Skip zero vectors

        stored_vector_norm = stored_vector / norm_stored

        # Calculate cosine similarity (dot product of normalized vectors)
        # similarity = 1 - cosine(live_vector, stored_vector_norm) # Using scipy
        similarity = np.dot(live_vector, stored_vector_norm)

        # Clip similarity to [-1, 1] just in case of floating point inaccuracies
        similarity = np.clip(similarity, -1.0, 1.0)

        if similarity > highest_similarity:
            highest_similarity = similarity
            potential_match_id = emp_id
            # Get name from DB, provide default if missing
            potential_match_name = data.get('name', f"Employee {emp_id}")

    # Check if the best match meets the threshold
    if highest_similarity >= MATCH_THRESHOLD:
        best_match_id = potential_match_id
        best_match_name = potential_match_name
    # else: # Keep name as "Unknown" if below threshold

    return best_match_id, best_match_name, highest_similarity


# --- Database Handling ---
def load_database(filepath):
    """Loads employee database from JSON file. Creates file if it doesn't exist."""
    if not os.path.exists(filepath):
        print(f"Database file '{filepath}' not found. Creating an empty one.")
        try:
            with open(filepath, 'w') as f:
                json.dump({}, f)
            return {}
        except IOError as e:
            print(f"Error: Could not create database file '{filepath}'. {e}")
            messagebox.showerror("Database Error", f"Could not create database file:\n{filepath}\n{e}")
            return None # Indicate failure

    try:
        with open(filepath, 'r') as f:
            database = json.load(f)
        # Basic validation
        valid_db = {}
        invalid_count = 0
        for emp_id, data in database.items():
            if isinstance(data, dict) and 'enhanced_feature' in data and \
               isinstance(data['enhanced_feature'], list) and \
               len(data['enhanced_feature']) == IDENTITY_VECTOR_DIM:
                valid_db[emp_id] = data
            else:
                 print(f"Warning: Invalid data format for Emp ID '{emp_id}' in DB. Skipping.")
                 invalid_count += 1
        if invalid_count > 0:
             messagebox.showwarning("Database Warning", f"{invalid_count} invalid entries found in the database and were skipped.")
        return valid_db
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filepath}'. File might be corrupted.")
        messagebox.showerror("Database Error", f"Could not decode JSON from '{filepath}'. File might be corrupted.")
        return None # Indicate failure
    except Exception as e:
        print(f"Error loading database: {e}")
        messagebox.showerror("Database Error", f"Error loading database:\n{e}")
        return None # Indicate failure

def save_database(filepath, database):
    """Saves employee database to JSON file."""
    try:
        # Optional: Backup before saving
        # if os.path.exists(filepath):
        #     shutil.copyfile(filepath, filepath + '.bak')
        with open(filepath, 'w') as f:
            json.dump(database, f, indent=4) # Use indent for readability
        return True
    except IOError as e:
        print(f"Error: Could not write to database file '{filepath}'. {e}")
        messagebox.showerror("Database Error", f"Could not write to database file:\n{filepath}\n{e}")
        return False
    except Exception as e:
        print(f"Error saving database: {e}")
        messagebox.showerror("Database Error", f"Error saving database:\n{e}")
        return False

# --- Core Logic: Registration Mode ---
def run_registration(employee_id, stop_event, status_callback):
    """Runs the registration process for a given employee ID."""
    status_callback(f"Initializing registration for ID: {employee_id}...")
    print(f"\n--- Starting Registration Mode for Employee ID: {employee_id} ---")

    # --- Initialization ---
    model = None
    pose = None
    cap = None
    employee_db = None

    try:
        # Load models
        status_callback("Loading YOLO model...")
        try: model = YOLO(MODEL_PATH)
        except Exception as e: raise RuntimeError(f"Cannot load YOLO model: {e}")

        status_callback("Initializing MediaPipe Pose...")
        try: pose = mp_pose.Pose( model_complexity=MP_POSE_MODEL_COMPLEXITY,
                                 min_detection_confidence=MP_POSE_MIN_DETECTION_CONFIDENCE,
                                 min_tracking_confidence=MP_POSE_MIN_TRACKING_CONFIDENCE,
                                 static_image_mode=False) # Use stream mode
        except Exception as e: raise RuntimeError(f"Cannot initialize MediaPipe Pose: {e}")

        # Load existing database (or create if needed)
        status_callback("Loading database...")
        employee_db = load_database(DB_FILE)
        if employee_db is None: raise RuntimeError("Failed to load or create the database.")

        # Init video capture
        status_callback(f"Opening video source: {VIDEO_SOURCE}...")
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open video source {VIDEO_SOURCE}.")
        w_cam = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h_cam = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Registration specific state
        registration_buffer = deque(maxlen=FEATURE_BUFFER_SIZE)
        saved_message = ""
        message_timer = 0

        status_callback(f"Registration ready for ID: {employee_id}. Press 'S' in window to save.")
        print("\nPlease position the employee clearly in the frame.")
        print("Ensure consistent movement (e.g., walk back and forth).")
        print(f"Buffer needs {FEATURE_BUFFER_SIZE} frames.")
        print("Press 'S' in the OpenCV window to calculate and save the feature vector.")
        print("Press 'Q' in the OpenCV window or close the main app to quit registration.")

        # --- Main Loop ---
        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("End of video source or camera error.")
                status_callback("Warning: End of video source or camera error.")
                time.sleep(1) # Avoid busy-looping if source ends suddenly
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- Detect the most prominent person ---
            # Simpler detection for registration: find largest 'person' box
            target_box = None
            largest_area = 0
            try:
                results = model(frame_rgb, classes=[0], verbose=False, conf=CONFIDENCE_THRESHOLD) # Class 0 is 'person' in COCO
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    for box_obj in results[0].boxes:
                         box = box_obj.xyxy.cpu().numpy().astype(int)[0]
                         # Basic validity check for box coordinates
                         if box[0] >= box[2] or box[1] >= box[3]: continue
                         area = (box[2] - box[1]) * (box[3] - box[0]) # Corrected area calculation
                         if area > largest_area:
                             largest_area = area
                             target_box = box
            except Exception as e:
                print(f"Error during YOLO detection: {e}")
                # Continue to next frame if detection fails

            # --- Process the target person ---
            if target_box is not None:
                x1, y1, x2, y2 = target_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), REG_BBOX_COLOR, 2)

                # Crop ROI for pose estimation with padding
                pad_x = int((x2 - x1) * 0.1); pad_y = int((y2 - y1) * 0.1)
                crop_x1 = max(0, x1 - pad_x); crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(w_cam, x2 + pad_x); crop_y2 = min(h_cam, y2 + pad_y)

                features = np.zeros(IDENTITY_VECTOR_DIM // 3, dtype=np.float32) # Initialize features
                landmarks_for_drawing = None # Reset landmarks for drawing

                if crop_x2 > crop_x1 and crop_y2 > crop_y1: # Ensure crop is valid
                    person_crop_rgb = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
                    try:
                        # Process with MediaPipe Pose
                        pose_results = pose.process(person_crop_rgb)

                        if pose_results.pose_landmarks:
                            landmarks_relative = pose_results.pose_landmarks # Landmarks relative to crop
                            features = extract_pose_features(landmarks_relative)

                            # Convert landmarks to absolute frame coordinates for drawing
                            crop_h, crop_w = person_crop_rgb.shape[:2]
                            landmarks_for_drawing = []
                            for lm in landmarks_relative.landmark:
                                abs_x = crop_x1 + lm.x * crop_w
                                abs_y = crop_y1 + lm.y * crop_h
                                landmarks_for_drawing.append({
                                    'x': abs_x, 'y': abs_y, 'z': lm.z,
                                    'visibility': lm.visibility
                                })

                            # Draw pose on the main frame if landmarks were found
                            if landmarks_for_drawing:
                                # Draw connections
                                for conn in mp_pose.POSE_CONNECTIONS:
                                    idx1, idx2 = conn
                                    if 0 <= idx1 < len(landmarks_for_drawing) and 0 <= idx2 < len(landmarks_for_drawing):
                                        lm1 = landmarks_for_drawing[idx1]; lm2 = landmarks_for_drawing[idx2]
                                        # Use a threshold for drawing connections too
                                        if lm1['visibility'] > MP_POSE_MIN_TRACKING_CONFIDENCE and \
                                           lm2['visibility'] > MP_POSE_MIN_TRACKING_CONFIDENCE:
                                            pt1 = (int(lm1['x']), int(lm1['y']))
                                            pt2 = (int(lm2['x']), int(lm2['y']))
                                            cv2.line(frame, pt1, pt2, POSE_CONNECTIONS_COLOR, 1)
                                # Draw landmark points
                                for lm in landmarks_for_drawing:
                                    if lm['visibility'] > MP_POSE_MIN_TRACKING_CONFIDENCE:
                                        cv2.circle(frame, (int(lm['x']), int(lm['y'])), 2, POSE_COLOR, -1)

                    except Exception as e:
                        print(f"Error processing pose for registration: {e}")
                        # Continue, features will be zeros

                # Add extracted features (or zeros if failed) to the buffer
                registration_buffer.append(features)
            # else: # No person detected in this frame
                # Optionally, add zeros to buffer to indicate no detection?
                # registration_buffer.append(np.zeros(IDENTITY_VECTOR_DIM // 3, dtype=np.float32))
                # Or just let the buffer fill slower when person not detected. Let's do the latter.
                # pass

            # --- Display Info & Handle Keys ---
            # Instructions Box
            box_h = 85 # Height of the info box
            cv2.rectangle(frame, (0, 0), (w_cam, box_h), REG_INFO_BOX_COLOR, -1) # Solid background
            cv2.putText(frame, f"REGISTERING ID: {employee_id}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, REG_TEXT_COLOR, 2)
            buffer_status = f"Buffer: {len(registration_buffer)}/{FEATURE_BUFFER_SIZE}"
            is_buffer_full = len(registration_buffer) == FEATURE_BUFFER_SIZE
            status_color = (0, 128, 0) if is_buffer_full else (0, 0, 128) # Green when full, Dark Red otherwise
            cv2.putText(frame, buffer_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, "Press 'S' to Save, 'Q' to Quit", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, REG_TEXT_COLOR, 2)

            # Display saved/error message temporarily
            if saved_message and time.time() < message_timer:
                 # Simple text overlay for message
                 text_size = cv2.getTextSize(saved_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                 text_x = (w_cam - text_size[0]) // 2
                 text_y = (h_cam + text_size[1]) // 2
                 cv2.putText(frame, saved_message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if "SAVED" in saved_message else (0,0,255), 2)
            elif saved_message and time.time() >= message_timer:
                 saved_message = "" # Clear message after duration

            cv2.imshow(f"Registration Mode - ID: {employee_id}", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Registration cancelled by user ('q' key).")
                status_callback(f"Registration for {employee_id} cancelled by user.")
                stop_event.set() # Signal stop
                break
            elif key == ord('s'):
                if len(registration_buffer) >= FEATURE_BUFFER_SIZE: # Ensure buffer is actually full
                    print(f"Calculating identity vector for {employee_id}...")
                    status_callback(f"Calculating identity vector for {employee_id}...")
                    id_vector = generate_identity_vector(registration_buffer)

                    if id_vector is not None:
                        print("Vector generated successfully. Saving to database...")
                        # Prepare entry (add default name/dept if not present)
                        if employee_id not in employee_db:
                             employee_db[employee_id] = {} # Create entry if new ID
                        # Store vector as list for JSON compatibility
                        employee_db[employee_id]['enhanced_feature'] = id_vector.tolist()
                        # Add default name/department if they don't exist for this ID
                        employee_db[employee_id].setdefault('name', f'Employee {employee_id}')
                        employee_db[employee_id].setdefault('department', 'Unknown')

                        if save_database(DB_FILE, employee_db):
                            print(f"Successfully saved/updated features for Employee ID: {employee_id}")
                            saved_message = f"SAVED ID: {employee_id}"
                            message_timer = time.time() + 3 # Show message for 3 seconds
                            status_callback(f"Successfully saved features for ID: {employee_id}")
                            registration_buffer.clear() # Clear buffer after successful save
                            # Keep running to allow re-saving if needed, or user can press 'q'
                        else:
                            print(f"Error: Failed to save database for Employee ID: {employee_id}")
                            saved_message = "DATABASE SAVE FAILED!"
                            message_timer = time.time() + 3
                            status_callback(f"Error: Failed to save database for ID: {employee_id}")
                    else:
                        print("Error: Could not generate identity vector. Ensure person is clearly visible and moving.")
                        saved_message = "Vector Generation Failed!"
                        message_timer = time.time() + 3
                        status_callback("Error: Could not generate identity vector.")
                else:
                    print(f"Buffer not full ({len(registration_buffer)}/{FEATURE_BUFFER_SIZE}). Cannot save yet.")
                    saved_message = "Buffer Not Full!"
                    message_timer = time.time() + 2 # Shorter message time
                    status_callback("Buffer not full. Cannot save yet.")

        # --- End of Loop ---

    except RuntimeError as e:
        print(f"Fatal Error during Registration setup: {e}")
        status_callback(f"Error: {e}")
        messagebox.showerror("Registration Error", str(e))
    except Exception as e:
        print(f"An unexpected error occurred during registration: {e}")
        traceback.print_exc()
        status_callback(f"Unexpected error: {e}")
        messagebox.showerror("Registration Error", f"An unexpected error occurred:\n{e}")
    finally:
        # --- Cleanup ---
        if cap: cap.release()
        if pose: pose.close()
        cv2.destroyAllWindows() # Close OpenCV windows specifically opened by this function
        print(f"--- Registration Mode for {employee_id} Ended ---")
        status_callback(f"Registration for {employee_id} finished.")


# --- Core Logic: Mapping Mode ---
def run_mapping(stop_event, status_callback):
    """Runs the real-time mapping and identification process."""
    status_callback("Initializing mapping mode...")
    print("\n--- Starting Mapping Mode ---")

    # --- Initialization ---
    model = None
    pose = None
    cap = None
    employee_db = None

    try:
        # Load models
        status_callback("Loading YOLO model...")
        try: model = YOLO(MODEL_PATH)
        except Exception as e: raise RuntimeError(f"Cannot load YOLO model: {e}")

        status_callback("Initializing MediaPipe Pose...")
        try: pose = mp_pose.Pose( model_complexity=MP_POSE_MODEL_COMPLEXITY,
                                 min_detection_confidence=MP_POSE_MIN_DETECTION_CONFIDENCE,
                                 min_tracking_confidence=MP_POSE_MIN_TRACKING_CONFIDENCE,
                                 static_image_mode=False) # Use stream mode
        except Exception as e: raise RuntimeError(f"Cannot initialize MediaPipe Pose: {e}")

        # Load database
        status_callback("Loading database...")
        employee_db = load_database(DB_FILE)
        if employee_db is None: raise RuntimeError("Failed to load the database. Mapping requires a database.")
        if not employee_db:
            print("Warning: Database is empty. Mapping mode will only show Track IDs.")
            status_callback("Warning: Database is empty. Only Track IDs will be shown.")
            # No need to raise error here, can proceed with just tracking

        # Init video capture
        status_callback(f"Opening video source: {VIDEO_SOURCE}...")
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open video source {VIDEO_SOURCE}.")
        w_cam = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h_cam = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Mapping specific state
        tracked_persons = {} # Key: track_id, Value: state dict
                             # State dict: {'buffer': deque, 'id_vec': np.array, 'emp_id': str,
                             #             'name': str, 'sim': float, 'last_seen': int,
                             #             'landmarks': list, 'bbox': tuple, 'last_matched_frame': int}
        frame_count = 0
        REMATCH_INTERVAL = FEATURE_BUFFER_SIZE * 2 # How often to re-run matching even if already identified

        status_callback("Mapping mode running...")
        print("Starting real-time mapping...")

        # --- Main Loop ---
        while cap.isOpened() and not stop_event.is_set():
            loop_start_time = time.time()
            success, frame = cap.read()
            if not success:
                print("End of video source or camera error.")
                status_callback("Warning: End of video source or camera error.")
                time.sleep(1) # Avoid busy-looping
                break
            frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- YOLO Tracking ---
            yolo_results = None
            current_tracked_ids = set()
            detections = []
            try:
                # Use tracker (persist=True)
                yolo_results = model.track(frame_rgb, persist=True, classes=[0], verbose=False,
                                          conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, tracker="bytetrack.yaml") # Or botsort.yaml

                if yolo_results and yolo_results[0].boxes is not None and yolo_results[0].boxes.id is not None:
                    boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)
                    confs = yolo_results[0].boxes.conf.cpu().numpy()

                    current_tracked_ids = set(track_ids)
                    for box, track_id, conf in zip(boxes, track_ids, confs):
                        # Validate box coordinates
                        if box[0] >= box[2] or box[1] >= box[3]: continue
                        detections.append({'box': box, 'track_id': track_id, 'conf': conf})
                # Handle cases where tracking might return results but no IDs (less common with persist=True)
                elif yolo_results and yolo_results[0].boxes is not None:
                    # Fallback to detection if IDs are missing (might happen on first frame or if tracker fails)
                     boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
                     confs = yolo_results[0].boxes.conf.cpu().numpy()
                     # Cannot assign track IDs here, could maybe use simple overlap logic if needed, but skip for now
                     # print("Warning: YOLO tracking results missing track IDs this frame.")
                     pass # Skip processing if no track IDs available

            except Exception as e:
                print(f"Error during YOLO tracking: {e}")
                traceback.print_exc() # More detailed error
                # Continue to next frame

            # --- Process Each Tracked Person ---
            for det in detections:
                track_id = det['track_id']
                box = det['box']
                x1, y1, x2, y2 = box

                # --- Update Track State ---
                if track_id not in tracked_persons:
                    tracked_persons[track_id] = {
                        'buffer': deque(maxlen=FEATURE_BUFFER_SIZE),
                        'id_vec': None,
                        'emp_id': None,
                        'name': 'Processing...', # Initial state
                        'sim': 0.0,
                        'last_seen': frame_count,
                        'landmarks': None,
                        'bbox': box,
                        'last_matched_frame': -REMATCH_INTERVAL # Ensure first match happens
                    }
                else:
                    # Update existing track
                    tracked_persons[track_id]['last_seen'] = frame_count
                    tracked_persons[track_id]['bbox'] = box # Update bbox coordinates


                # --- Crop, Pose Estimation, Feature Extraction ---
                # Crop ROI with padding
                pad_x = int((x2 - x1) * 0.1); pad_y = int((y2 - y1) * 0.1)
                crop_x1 = max(0, x1 - pad_x); crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(w_cam, x2 + pad_x); crop_y2 = min(h_cam, y2 + pad_y)

                features = np.zeros(IDENTITY_VECTOR_DIM // 3, dtype=np.float32) # Default features
                landmarks_for_drawing = None

                if crop_x2 > crop_x1 and crop_y2 > crop_y1: # Check for valid crop dimensions
                    person_crop_rgb = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
                    try:
                        # Process with MediaPipe Pose
                        pose_results = pose.process(person_crop_rgb)
                        if pose_results.pose_landmarks:
                            landmarks_relative = pose_results.pose_landmarks
                            features = extract_pose_features(landmarks_relative)

                            # Convert landmarks to absolute frame coordinates for drawing
                            crop_h, crop_w = person_crop_rgb.shape[:2]
                            landmarks_for_drawing = []
                            for lm in landmarks_relative.landmark:
                                abs_x = crop_x1 + lm.x * crop_w
                                abs_y = crop_y1 + lm.y * crop_h
                                landmarks_for_drawing.append({
                                    'x': abs_x, 'y': abs_y, 'z': lm.z,
                                    'visibility': lm.visibility
                                })
                    except Exception as e:
                        print(f"Error processing pose for track {track_id}: {e}")
                        # Features remain zeros

                # Update track data
                tracked_persons[track_id]['landmarks'] = landmarks_for_drawing
                tracked_persons[track_id]['buffer'].append(features)

                # --- Generate ID Vector & Match (if buffer full and conditions met) ---
                buffer = tracked_persons[track_id]['buffer']
                can_match = len(buffer) == FEATURE_BUFFER_SIZE
                should_rematch = (frame_count - tracked_persons[track_id]['last_matched_frame']) >= REMATCH_INTERVAL

                if can_match and (tracked_persons[track_id]['emp_id'] is None or should_rematch):
                    id_vec = generate_identity_vector(buffer)
                    if id_vec is not None:
                        tracked_persons[track_id]['id_vec'] = id_vec # Store the generated vector
                        emp_id, name, similarity = match_identity(id_vec, employee_db)
                        tracked_persons[track_id]['last_matched_frame'] = frame_count # Record match attempt time

                        # Update track identity based on match result
                        if emp_id is not None: # Found a match above threshold
                            tracked_persons[track_id]['emp_id'] = emp_id
                            tracked_persons[track_id]['name'] = name
                            tracked_persons[track_id]['sim'] = similarity
                        else: # No match above threshold
                            # If previously identified, maybe keep it for a while? Or reset? Reset for now.
                            tracked_persons[track_id]['emp_id'] = None
                            tracked_persons[track_id]['name'] = "Unknown"
                            tracked_persons[track_id]['sim'] = similarity # Show highest similarity even if low
                    else:
                        # Failed to generate ID vector, keep previous state but maybe log?
                        # print(f"Track {track_id}: Failed to generate ID vector from buffer.")
                        # Don't update last_matched_frame if generation failed
                        pass


            # --- Visualization & Cleanup Old Tracks ---
            tracks_to_remove = []
            for track_id, data in tracked_persons.items():
                # Remove tracks that haven't been seen for a while
                # Use a longer timeout for removal than the rematch interval
                if frame_count - data['last_seen'] > FEATURE_BUFFER_SIZE * 5:
                    tracks_to_remove.append(track_id)
                    continue

                # Draw only currently visible tracks
                if track_id in current_tracked_ids:
                    x1, y1, x2, y2 = data['bbox']
                    emp_id = data['emp_id']; name = data['name']; sim = data['sim']
                    buffer_len = len(data['buffer'])

                    # Determine BBox color based on state
                    color = MAP_BBOX_COLOR_UNKNOWN # Default: Unknown / Ready to match
                    if emp_id is not None:
                        color = MAP_BBOX_COLOR_KNOWN # Identified
                    elif buffer_len == FEATURE_BUFFER_SIZE:
                        color = MAP_BBOX_COLOR_UNKNOWN # Buffer full, but no match / unknown
                    elif buffer_len > 0 :
                        color = MAP_BBOX_COLOR_PROCESSING # Buffer filling

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Text Label Construction
                    id_label = f"ID: {emp_id}" if emp_id else f"Track: {track_id}"
                    name_label = f"{name}"
                    # Show similarity only when a match attempt was made (buffer full)
                    status_label = f"Sim: {sim:.2f}" if buffer_len == FEATURE_BUFFER_SIZE else f"Buf: {buffer_len}/{FEATURE_BUFFER_SIZE}"
                    label_text = f"{id_label} | {name_label} | {status_label}"

                    # Position text above the box
                    (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    text_y = y1 - 10 if y1 > 25 else y1 + label_height + 5 # Adjust if box is near top
                    text_x = x1

                    # Add a background rectangle for the text for better visibility
                    cv2.rectangle(frame, (text_x, text_y - label_height - baseline), (text_x + label_width, text_y + baseline), color, -1)
                    cv2.putText(frame, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR if color != MAP_BBOX_COLOR_PROCESSING else (0,0,0), 1, cv2.LINE_AA) # Black text on Cyan

                    # Draw Pose Landmarks if enabled and available
                    if DRAW_POSE_LANDMARKS and data['landmarks']:
                        landmarks = data['landmarks']
                        # Draw connections
                        for conn in mp_pose.POSE_CONNECTIONS:
                            idx1, idx2 = conn
                            if 0 <= idx1 < len(landmarks) and 0 <= idx2 < len(landmarks):
                                lm1, lm2 = landmarks[idx1], landmarks[idx2]
                                if lm1['visibility'] > MP_POSE_MIN_TRACKING_CONFIDENCE and lm2['visibility'] > MP_POSE_MIN_TRACKING_CONFIDENCE:
                                    pt1 = (int(lm1['x']), int(lm1['y']))
                                    pt2 = (int(lm2['x']), int(lm2['y']))
                                    cv2.line(frame, pt1, pt2, POSE_CONNECTIONS_COLOR, 1)
                        # Draw points
                        for lm in landmarks:
                             if lm['visibility'] > MP_POSE_MIN_TRACKING_CONFIDENCE:
                                 cv2.circle(frame, (int(lm['x']), int(lm['y'])), 2, POSE_COLOR, -1)

            # Remove old tracks
            for track_id in tracks_to_remove:
                if track_id in tracked_persons:
                    # print(f"Removing inactive track: {track_id}")
                    del tracked_persons[track_id]

            # --- Display FPS & Frame ---
            loop_end_time = time.time()
            processing_time = loop_end_time - loop_start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            cv2.putText(frame, f"MAP MODE | FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA) # Black outline
            cv2.putText(frame, f"MAP MODE | FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA) # White text
            cv2.imshow("Employee Mapping Mode", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Mapping mode stopped by user ('q' key).")
                status_callback("Mapping stopped by user.")
                stop_event.set() # Signal stop
                break

        # --- End of Loop ---

    except RuntimeError as e:
        print(f"Fatal Error during Mapping setup: {e}")
        status_callback(f"Error: {e}")
        messagebox.showerror("Mapping Error", str(e))
    except Exception as e:
        print(f"An unexpected error occurred during mapping: {e}")
        traceback.print_exc()
        status_callback(f"Unexpected error: {e}")
        messagebox.showerror("Mapping Error", f"An unexpected error occurred:\n{e}")
    finally:
        # --- Cleanup ---
        if cap: cap.release()
        if pose: pose.close()
        cv2.destroyAllWindows() # Close OpenCV windows opened by this function
        print("--- Mapping Mode Ended ---")
        status_callback("Mapping finished.")


# --- Tkinter GUI Application ---
class TkinterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Employee Pose ID System")
        self.root.geometry("450x250") # Adjusted size

        self.current_mode = None # 'register' or 'map'
        self.mode_thread = None
        self.stop_event = threading.Event()

        # Styling
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", font=('Helvetica', 10))
        style.configure("TLabel", padding=5, font=('Helvetica', 10))
        style.configure("TEntry", padding=5)
        style.configure("Status.TLabel", font=('Helvetica', 9), relief="sunken", anchor="w")

        # Main Frame
        main_frame = ttk.Frame(root, padding="10 10 10 10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # Registration Section
        reg_frame = ttk.LabelFrame(main_frame, text="Registration", padding="10 10 10 10")
        reg_frame.pack(pady=5, fill=tk.X)

        ttk.Label(reg_frame, text="Employee ID:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.emp_id_entry = ttk.Entry(reg_frame, width=20)
        self.emp_id_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.register_button = ttk.Button(reg_frame, text="Start Registration", command=self._start_registration)
        self.register_button.grid(row=0, column=2, padx=5, pady=5)

        # Mapping Section
        map_frame = ttk.LabelFrame(main_frame, text="Mapping", padding="10 10 10 10")
        map_frame.pack(pady=5, fill=tk.X)

        self.map_button = ttk.Button(map_frame, text="Start Mapping", command=self._start_mapping)
        self.map_button.pack(padx=5, pady=5) # Centered in its frame

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready.")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, style="Status.TLabel")
        status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        # Stop Button (appears when a mode is running)
        self.stop_button = ttk.Button(main_frame, text="Stop Current Process", command=self._stop_process, state=tk.DISABLED)
        self.stop_button.pack(side=tk.BOTTOM, pady=5)


        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _update_status(self, message):
        """Safely update the status bar from any thread."""
        self.root.after(0, self.status_var.set, message)

    def _set_buttons_state(self, state):
        """Enable/Disable main control buttons."""
        self.register_button.config(state=state)
        self.map_button.config(state=state)
        self.emp_id_entry.config(state=state) # Disable entry during processing
        # Enable/disable stop button inversely
        self.stop_button.config(state=tk.NORMAL if state == tk.DISABLED else tk.DISABLED)


    def _start_registration(self):
        emp_id = self.emp_id_entry.get().strip()
        if not emp_id:
            messagebox.showerror("Input Error", "Please enter an Employee ID.")
            return

        if self.mode_thread and self.mode_thread.is_alive():
            messagebox.showwarning("Busy", "Another process is already running.")
            return

        self.current_mode = 'register'
        self.stop_event.clear() # Reset stop event
        self._set_buttons_state(tk.DISABLED)
        self._update_status(f"Starting registration for ID: {emp_id}...")

        # Run registration in a separate thread
        self.mode_thread = threading.Thread(target=self._run_mode_thread,
                                            args=(run_registration, emp_id),
                                            daemon=True) # Daemon allows closing app even if thread hangs (use carefully)
        self.mode_thread.start()

    def _start_mapping(self):
        if self.mode_thread and self.mode_thread.is_alive():
            messagebox.showwarning("Busy", "Another process is already running.")
            return

        self.current_mode = 'map'
        self.stop_event.clear() # Reset stop event
        self._set_buttons_state(tk.DISABLED)
        self._update_status("Starting mapping mode...")

        # Run mapping in a separate thread
        self.mode_thread = threading.Thread(target=self._run_mode_thread,
                                            args=(run_mapping,),
                                            daemon=True)
        self.mode_thread.start()

    def _run_mode_thread(self, target_func, *args):
        """Wrapper to run the target function and handle completion."""
        try:
            # Pass the stop event and status callback to the target function
            target_func(*args, stop_event=self.stop_event, status_callback=self._update_status)
        except Exception as e:
            print(f"Error in {self.current_mode} thread: {e}")
            traceback.print_exc()
            # Update status via callback to ensure thread safety
            self._update_status(f"Error during {self.current_mode}: {e}")
            # Show error in GUI thread
            self.root.after(0, messagebox.showerror, "Runtime Error", f"An error occurred during {self.current_mode}:\n{e}")
        finally:
            # Ensure GUI elements are re-enabled in the main thread
            self.root.after(0, self._on_mode_complete)

    def _on_mode_complete(self):
        """Called after a mode finishes execution (success or error)."""
        # Final status update might be set by the thread itself, or set a generic one here
        final_status = self.status_var.get() # Keep the last status from the thread
        if "Error" not in final_status and "cancelled" not in final_status:
             self._update_status(f"{self.current_mode.capitalize()} finished. Ready.")
        elif "cancelled" in final_status:
             self._update_status(f"{self.current_mode.capitalize()} cancelled. Ready.")
        # else keep the error message
        self.current_mode = None
        self.mode_thread = None
        self._set_buttons_state(tk.NORMAL) # Re-enable buttons

    def _stop_process(self):
        """Signals the running process to stop."""
        if self.mode_thread and self.mode_thread.is_alive():
            self._update_status(f"Attempting to stop {self.current_mode}...")
            self.stop_event.set() # Signal the thread to stop
            # Button state will be reset by _on_mode_complete when the thread actually exits
        else:
            self._update_status("No process running.")

    def _on_closing(self):
        """Handles the event when the main window is closed."""
        if self.mode_thread and self.mode_thread.is_alive():
            if messagebox.askokcancel("Quit", "A process is running. Stop it and exit?"):
                self._update_status("Stopping process and exiting...")
                self.stop_event.set()
                # Wait briefly for the thread to potentially acknowledge stop?
                # self.mode_thread.join(timeout=1.0) # Careful with join on main thread
                self.root.destroy()
            else:
                return # Don't close if user cancels
        else:
            self.root.destroy()

    def run(self):
        """Starts the Tkinter main loop."""
        self.root.mainloop()


# --- Main Entry Point ---
if __name__ == "__main__":
    # Removed argparse logic
    root = tk.Tk()
    app = TkinterApp(root)
    app.run()