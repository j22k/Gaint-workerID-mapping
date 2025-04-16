import cv2
import mediapipe as mp
import numpy as np
import json
import os
from collections import deque
import time
import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread
from PIL import Image, ImageTk
import logging
from scipy.signal import find_peaks
from scipy.stats import entropy
from scipy import fft
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ultralytics import YOLO  # Import YOLO
import torch # PyTorch needed by YOLO

# -------------------------------
# Configuration and Settings
# -------------------------------
EMP_DB_FILE = "employee_multi_enhanced_db.json" # Database file
LOG_FILE = "warehouse_multi_tracking.log"      # Log file
ACTIVITY_LOG_FILE = "warehouse_multi_activity_log.csv" # Employee activity log

# --- Tracking & Feature Settings ---
FEATURE_BUFFER_LENGTH = 45
MIN_DETECTION_FRAMES = 10
MAX_PEOPLE_TRACK = 15 # Increased for multi-person
MATCH_THRESHOLD = 0.70 # Cosine similarity threshold
FEATURE_DIMENSION = 54
REGISTRATION_CENTER_THRESHOLD_FACTOR = 0.25
IOU_THRESHOLD = 0.4 # IoU for tracker association
TRACKER_TIMEOUT_SEC = 3.0 # Seconds without seeing a tracker before removing it

# --- YOLOv8 Settings ---
YOLO_MODEL_PATH = 'yolov8n.pt' # Nano model - fast but less accurate. Use yolov8s.pt or yolov8m.pt for better accuracy.
PERSON_CLASS_ID = 0 # In COCO dataset, 'person' is class 0
PERSON_CONF_THRESHOLD = 0.45 # Minimum confidence for YOLO person detection

# --- Performance ---
# Set to True if GPU is available and PyTorch is installed with CUDA support
USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if USE_GPU else 'cpu'

# -------------------------------
# Logging Setup (Same as before)
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logging.info(f"Application started. Using device: {DEVICE}")

# -------------------------------
# Helper Functions (Same as before)
# -------------------------------
def compute_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return 0.0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_person_bbox_from_landmarks(landmarks, frame_shape): # Renamed for clarity
    h, w = frame_shape[:2]
    # Use landmarks with visibility > 0.5 for bounding box calculation
    visible_landmarks = [lm for lm in landmarks if lm.visibility > 0.5]
    if not visible_landmarks:
        # This shouldn't happen if called after successful pose estimation, but as fallback:
        return 0, 0, w, h, w // 2, h // 2

    # Use actual landmark coordinates (relative to full frame now)
    x_coords = [lm.x * w for lm in visible_landmarks]
    y_coords = [lm.y * h for lm in visible_landmarks]

    padding_factor = 0.1
    x_min = max(0, int(min(x_coords) - padding_factor * (max(x_coords) - min(x_coords))))
    y_min = max(0, int(min(y_coords) - padding_factor * (max(y_coords) - min(y_coords))))
    x_max = min(w, int(max(x_coords) + padding_factor * (max(x_coords) - min(x_coords))))
    y_max = min(h, int(max(y_coords) + padding_factor * (max(y_coords) - min(y_coords))))

    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # Return format consistent with tracker needs
    return (x_min, y_min, x_max, y_max, center_x, center_y)


def check_overlap(bbox1, bbox2, iou_threshold=IOU_THRESHOLD):
    x1_min, y1_min, x1_max, y1_max = bbox1[:4] # Tracker bbox might be slightly different
    x2_min, y2_min, x2_max, y2_max = bbox2[:4] # Detection bbox from YOLO

    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    if x_right < x_left or y_bottom < y_top:
        return False, 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = float(box1_area + box2_area - intersection_area)

    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou > iou_threshold, iou

def log_employee_activity(emp_id, action="detected"): # Same as before
    try:
        file_exists = os.path.exists(ACTIVITY_LOG_FILE)
        with open(ACTIVITY_LOG_FILE, 'a') as f:
            if not file_exists:
                f.write("timestamp,employee_id,action\n")
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp},{emp_id},{action}\n")
    except Exception as e:
        logging.error(f"Failed to log activity for {emp_id}: {e}")

# -------------------------------
# MediaPipe Pose Detector Class (Same as before)
# -------------------------------
class PoseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1):
        self.mp_pose = mp.solutions.pose
        # Initialize for single person detection (will be run on crops)
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame): # Now expects a single person crop
        """Process a frame crop to detect pose landmarks."""
        # Ensure frame is writeable before color conversion if needed
        if not frame.flags.writeable:
           frame = frame.copy() # Make a writeable copy if it's read-only

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Return landmarks if found for the person in the crop
        return results.pose_landmarks # Returns the landmarks object or None

    def draw_landmarks(self, frame, landmarks): # Same drawing logic
        self.mp_drawing.draw_landmarks(
            frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        return frame

    def close(self):
        self.pose.close()


# ---------------------------------------
# Enhanced Feature Extractor (Unchanged)
# ---------------------------------------
# ... (Keep the full EnhancedFeatureExtractor class from the previous version here) ...
class EnhancedFeatureExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        # If using normalization:
        # self.scaler = StandardScaler()
        # self.scaler_fitted = False

    def _get_landmark(self, landmarks, landmark_enum, frame_shape):
        """Helper to get landmark coordinates safely."""
        h, w = frame_shape[:2]
        try:
            # Access landmark directly if landmarks is the list/iterable
            # Adjust if landmarks is the full results object
            lm = landmarks[landmark_enum.value]
            if lm.visibility < 0.3: # Threshold for considering a landmark visible
                return None
            # Return pixel coordinates relative to the crop initially
            return np.array([lm.x * w, lm.y * h])
        except IndexError:
            logging.debug(f"Landmark {landmark_enum.name} index out of bounds.")
            return None
        except TypeError: # If landmarks is None or not subscriptable
            logging.debug(f"Invalid landmarks object passed to _get_landmark for {landmark_enum.name}")
            return None
        except AttributeError: # If visibility attribute missing
             logging.debug(f"Attribute error accessing landmark {landmark_enum.name}")
             return None


    def extract_joint_angles(self, landmarks, frame_shape):
        """Extract comprehensive set of joint angles."""
        coords = {}
        required_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.NOSE
        ]

        # Use the provided landmarks directly (assuming they are already translated and normalized)
        h, w = frame_shape[:2] # Use full frame shape here

        valid_coords = True
        for lm_enum in required_landmarks:
            try:
                lm = landmarks[lm_enum.value]
                # Use already normalized coordinates
                if lm.visibility >= 0.3:
                     coords[lm_enum] = np.array([lm.x * w, lm.y * h])
                else:
                     coords[lm_enum] = None
                     valid_coords = False
            except (IndexError, TypeError, AttributeError):
                 coords[lm_enum] = None
                 valid_coords = False
                 logging.debug(f"Could not get valid coordinate for {lm_enum.name}")


        if not valid_coords:
            logging.debug("Missing some required landmarks for angle calculation after check.")
            return np.zeros(8)

        angles = []
        try:
            # Lower body angles
            angles.append(compute_angle(coords[self.mp_pose.PoseLandmark.RIGHT_HIP], coords[self.mp_pose.PoseLandmark.RIGHT_KNEE], coords[self.mp_pose.PoseLandmark.RIGHT_ANKLE]))
            angles.append(compute_angle(coords[self.mp_pose.PoseLandmark.LEFT_HIP], coords[self.mp_pose.PoseLandmark.LEFT_KNEE], coords[self.mp_pose.PoseLandmark.LEFT_ANKLE]))
            # Hip angles
            angles.append(compute_angle(coords[self.mp_pose.PoseLandmark.RIGHT_SHOULDER], coords[self.mp_pose.PoseLandmark.RIGHT_HIP], coords[self.mp_pose.PoseLandmark.RIGHT_KNEE]))
            angles.append(compute_angle(coords[self.mp_pose.PoseLandmark.LEFT_SHOULDER], coords[self.mp_pose.PoseLandmark.LEFT_HIP], coords[self.mp_pose.PoseLandmark.LEFT_KNEE]))
            # Upper body angles
            angles.append(compute_angle(coords[self.mp_pose.PoseLandmark.RIGHT_SHOULDER], coords[self.mp_pose.PoseLandmark.RIGHT_ELBOW], coords[self.mp_pose.PoseLandmark.RIGHT_WRIST]))
            angles.append(compute_angle(coords[self.mp_pose.PoseLandmark.LEFT_SHOULDER], coords[self.mp_pose.PoseLandmark.LEFT_ELBOW], coords[self.mp_pose.PoseLandmark.LEFT_WRIST]))
            # Spine angle approximation
            mid_shoulder = (coords[self.mp_pose.PoseLandmark.LEFT_SHOULDER] + coords[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]) / 2 if coords[self.mp_pose.PoseLandmark.LEFT_SHOULDER] is not None and coords[self.mp_pose.PoseLandmark.RIGHT_SHOULDER] is not None else None
            mid_hip = (coords[self.mp_pose.PoseLandmark.LEFT_HIP] + coords[self.mp_pose.PoseLandmark.RIGHT_HIP]) / 2 if coords[self.mp_pose.PoseLandmark.LEFT_HIP] is not None and coords[self.mp_pose.PoseLandmark.RIGHT_HIP] is not None else None
            neck = mid_shoulder + (coords[self.mp_pose.PoseLandmark.NOSE] - mid_shoulder) * 0.1 if mid_shoulder is not None and coords[self.mp_pose.PoseLandmark.NOSE] is not None else None

            if neck is not None and mid_shoulder is not None and mid_hip is not None:
                 angles.append(compute_angle(neck, mid_shoulder, mid_hip))
            else:
                 angles.append(0.0) # Default if points missing

            # Torso twist approximation
            if coords[self.mp_pose.PoseLandmark.RIGHT_SHOULDER] is not None and coords[self.mp_pose.PoseLandmark.LEFT_SHOULDER] is not None and coords[self.mp_pose.PoseLandmark.RIGHT_HIP] is not None and coords[self.mp_pose.PoseLandmark.LEFT_HIP] is not None:
                 shoulder_vec = coords[self.mp_pose.PoseLandmark.RIGHT_SHOULDER] - coords[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                 hip_vec = coords[self.mp_pose.PoseLandmark.RIGHT_HIP] - coords[self.mp_pose.PoseLandmark.LEFT_HIP]
                 angles.append(abs(shoulder_vec[0] - hip_vec[0]))
            else:
                 angles.append(0.0) # Default

        except Exception as e:
             logging.error(f"Error calculating angles: {e}")
             return np.zeros(8) # Return zero vector on error

        return np.nan_to_num(np.array(angles))

    def extract_body_proportions(self, landmarks, frame_shape):
        """Extract body proportion features."""
        coords = {}
        required_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]

        h, w = frame_shape[:2]
        valid_coords = True
        for lm_enum in required_landmarks:
             try:
                lm = landmarks[lm_enum.value]
                if lm.visibility >= 0.3:
                     coords[lm_enum] = np.array([lm.x * w, lm.y * h])
                else:
                     coords[lm_enum] = None
                     valid_coords = False
             except (IndexError, TypeError, AttributeError):
                 coords[lm_enum] = None
                 valid_coords = False
                 logging.debug(f"Could not get valid coordinate for proportion: {lm_enum.name}")


        if not valid_coords:
             logging.debug("Missing landmarks for proportion calculation.")
             return np.zeros(3)

        proportions = np.zeros(3)
        try:
            # Calculate distances (only if points are valid)
            shoulder_width = np.linalg.norm(coords[self.mp_pose.PoseLandmark.RIGHT_SHOULDER] - coords[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
            hip_width = np.linalg.norm(coords[self.mp_pose.PoseLandmark.RIGHT_HIP] - coords[self.mp_pose.PoseLandmark.LEFT_HIP])
            right_upper_leg = np.linalg.norm(coords[self.mp_pose.PoseLandmark.RIGHT_HIP] - coords[self.mp_pose.PoseLandmark.RIGHT_KNEE])
            left_upper_leg = np.linalg.norm(coords[self.mp_pose.PoseLandmark.LEFT_HIP] - coords[self.mp_pose.PoseLandmark.LEFT_KNEE])
            right_lower_leg = np.linalg.norm(coords[self.mp_pose.PoseLandmark.RIGHT_KNEE] - coords[self.mp_pose.PoseLandmark.RIGHT_ANKLE])
            left_lower_leg = np.linalg.norm(coords[self.mp_pose.PoseLandmark.LEFT_KNEE] - coords[self.mp_pose.PoseLandmark.LEFT_ANKLE])
            mid_shoulder = (coords[self.mp_pose.PoseLandmark.LEFT_SHOULDER] + coords[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]) / 2
            mid_hip = (coords[self.mp_pose.PoseLandmark.LEFT_HIP] + coords[self.mp_pose.PoseLandmark.RIGHT_HIP]) / 2
            torso_length = np.linalg.norm(mid_shoulder - mid_hip)

            # Calculate ratios
            epsilon = 1e-6
            proportions[0] = shoulder_width / (hip_width + epsilon)
            avg_leg_length = (right_upper_leg + left_upper_leg + right_lower_leg + left_lower_leg) / 2
            proportions[1] = avg_leg_length / (torso_length + epsilon)
            avg_upper_leg = (right_upper_leg + left_upper_leg) / 2
            avg_lower_leg = (right_lower_leg + left_lower_leg) / 2
            proportions[2] = avg_upper_leg / (avg_lower_leg + epsilon)

        except Exception as e:
             logging.error(f"Error calculating proportions: {e}")
             # Proportions remains zeros

        return np.nan_to_num(proportions)

    def extract_movement_dynamics(self, pose_history):
        """Extract dynamic movement features from pose history."""
        history_len = len(pose_history)
        if history_len < 10:
            return np.zeros(7)

        dynamics = np.zeros(7)
        try:
            # Extract using the landmark value (index) - Ensure pose_history stores landmark lists/iterables
            left_ankle_y = np.array([pose[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y for pose in pose_history if pose and len(pose)>self.mp_pose.PoseLandmark.LEFT_ANKLE.value])
            right_ankle_y = np.array([pose[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y for pose in pose_history if pose and len(pose)>self.mp_pose.PoseLandmark.RIGHT_ANKLE.value])

            if len(left_ankle_y) != history_len or len(right_ankle_y) != history_len:
                 logging.warning("Inconsistent data in pose history for dynamics.")
                 return dynamics # Return zeros

            avg_ankle_y = (left_ankle_y + right_ankle_y) / 2.0

            velocity = np.gradient(avg_ankle_y)
            acceleration = np.gradient(velocity)

            # FFT requires real frequencies, need estimated FPS
            # Placeholder: use index-based frequency approx
            fft_result = np.abs(fft.fft(avg_ankle_y - np.mean(avg_ankle_y)))
            freqs = fft.fftfreq(history_len)
            valid_range = (freqs > 0.01) & (freqs < 0.3) # Heuristic range
            dominant_freq_idx = np.argmax(fft_result[valid_range]) if np.any(valid_range) else 0
            stride_frequency = freqs[valid_range][dominant_freq_idx] if dominant_freq_idx > 0 and len(freqs[valid_range]) > dominant_freq_idx else 0.0

            jerk = np.gradient(acceleration)
            smoothness = 1.0 / (np.mean(np.abs(jerk)) + 1e-6)

            hist, _ = np.histogram(velocity, bins=10, density=True)
            movement_entropy = entropy(hist)

            dynamics = np.array([
                np.mean(np.abs(velocity)), np.std(velocity),
                np.mean(np.abs(acceleration)), np.std(acceleration),
                stride_frequency,
                smoothness,
                movement_entropy
            ])
        except (IndexError, TypeError, AttributeError, ValueError) as e:
             logging.warning(f"Error extracting dynamics: {e}. Returning zeros.")
             # Dynamics remains zeros

        return np.nan_to_num(dynamics)

    def extract_combined_features(self, landmarks, frame_shape, pose_history=None):
        """Extract and combine all features."""
        # Landmarks should be the list/iterable of landmark objects (normalized, full frame)
        joint_angles = self.extract_joint_angles(landmarks, frame_shape) # 8 features
        body_proportions = self.extract_body_proportions(landmarks, frame_shape) # 3 features

        if pose_history is not None:
            dynamics = self.extract_movement_dynamics(pose_history) # 7 features
        else:
            dynamics = np.zeros(7)

        combined = np.concatenate([joint_angles, body_proportions, dynamics])
        return np.nan_to_num(combined) # Shape (18,) per frame

    def process_feature_buffer(self, feature_buffer):
        """Process a buffer of per-frame features into a single identity vector."""
        if len(feature_buffer) < FEATURE_BUFFER_LENGTH // 2:
            return None

        features = np.array(feature_buffer) # Shape: (buffer_len, 18)

        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
             logging.warning("NaN or Inf detected in feature buffer, replacing with 0.")
             features = np.nan_to_num(features)

        mean_features = np.mean(features, axis=0)
        std_features = np.std(features, axis=0)
        median_features = np.median(features, axis=0)

        # Concatenate stats: 18 means + 18 stds + 18 medians = 54 features
        processed = np.concatenate([mean_features, std_features, median_features])

        # Ensure consistent dimensionality
        current_dim = len(processed)
        if current_dim != FEATURE_DIMENSION:
             logging.warning(f"Processed feature dimension ({current_dim}) mismatch expected ({FEATURE_DIMENSION}). Adjusting.")
             if current_dim > FEATURE_DIMENSION:
                 processed = processed[:FEATURE_DIMENSION]
             else:
                 padding = np.zeros(FEATURE_DIMENSION - current_dim)
                 processed = np.concatenate([processed, padding])

        return processed.astype(np.float32) # Ensure float32

    def match_features(self, feature_vector, template_feature, threshold=MATCH_THRESHOLD):
        """Match features using cosine similarity."""
        feature_norm = np.linalg.norm(feature_vector)
        template_norm = np.linalg.norm(template_feature)

        if feature_norm == 0 or template_norm == 0: return 0.0

        similarity = np.dot(feature_vector, template_feature) / (feature_norm * template_norm)
        similarity = np.clip(similarity, 0.0, 1.0)

        return similarity if similarity >= threshold else 0.0


# ---------------------------------------
# Enhanced Person Tracker (Unchanged structure, but how methods are called changes)
# ---------------------------------------
# ... (Keep the full EnhancedPersonTracker class from the previous version here) ...
class EnhancedPersonTracker:
    def __init__(self, track_id, initial_bbox): # Takes YOLO bbox initially
        self.track_id = track_id
        # Store the *detector's* bounding box initially, might be updated by pose bbox later if desired
        self.bbox = initial_bbox # (x_min, y_min, x_max, y_max, cx, cy)
        self.feature_buffer = deque(maxlen=FEATURE_BUFFER_LENGTH)
        # Stores the translated, normalized landmark lists/iterables
        self.pose_history = deque(maxlen=FEATURE_BUFFER_LENGTH)
        self.last_matched_id = None
        self.match_confidence = 0.0
        self.consecutive_matches = 0
        self.last_seen = time.time()
        self.stable = False
        self.employee_info = None
        self.feature_extractor = EnhancedFeatureExtractor()
        self.processed_feature = None
        self.last_bbox = initial_bbox # Keep track of last known bbox

    def update_position(self, bbox): # Update with new bbox from matched detection
        self.last_bbox = self.bbox # Store previous before update
        self.bbox = bbox
        self.last_seen = time.time()

    def add_landmarks(self, landmarks_list, frame_shape): # Expects translated, normalized landmark list
        """Add translated pose landmarks, extract features."""
        # Landmarks_list is the list/iterable of landmark objects (e.g., from results.pose_landmarks.landmark)
        if landmarks_list:
            self.pose_history.append(landmarks_list)

            # Extract features using the translated/normalized landmarks
            current_features = self.feature_extractor.extract_combined_features(
                landmarks_list, frame_shape,
                self.pose_history if len(self.pose_history) > 5 else None
            )
            if current_features is not None:
                self.feature_buffer.append(current_features)

            # Update the processed feature if buffer is sufficiently full
            if len(self.feature_buffer) >= FEATURE_BUFFER_LENGTH // 2:
                 self.processed_feature = self.feature_extractor.process_feature_buffer(self.feature_buffer)
        else:
             logging.warning(f"Tracker {self.track_id}: Received empty landmarks list in add_landmarks.")


    def get_identity_feature(self):
        """Return the latest processed identity feature vector."""
        return self.processed_feature

    def match_with_database(self, emp_db):
        """Match current feature vector with employee database."""
        identity_feature = self.get_identity_feature()

        if identity_feature is None:
            self.reset_match_state()
            return None, 0.0, None

        best_match_id = None
        best_confidence = 0.0

        for emp_id, emp_data in emp_db.employees.items():
            if 'enhanced_feature' in emp_data and isinstance(emp_data['enhanced_feature'], np.ndarray):
                template = emp_data['enhanced_feature']
                if template.shape == identity_feature.shape:
                    # Use match_features directly (threshold is applied inside)
                    similarity = self.feature_extractor.match_features(
                        identity_feature, template
                    )
                    # If similarity > 0, it means it met the threshold
                    if similarity > best_confidence:
                        best_confidence = similarity # Store the actual similarity
                        best_match_id = emp_id
                else:
                    logging.warning(f"Shape mismatch for Emp {emp_id}: Template {template.shape}, Current {identity_feature.shape}")

        # Update tracker state based on match result
        if best_match_id is not None: # Found a match above threshold
            if best_match_id == self.last_matched_id:
                self.consecutive_matches += 1
            else:
                self.last_matched_id = best_match_id
                # Store the actual confidence score, not just threshold flag
                self.match_confidence = best_confidence
                self.consecutive_matches = 1
                self.stable = False
                self.employee_info = None

            if not self.stable and self.consecutive_matches >= MIN_DETECTION_FRAMES:
                self.stable = True
                self.employee_info = emp_db.get_employee_info(best_match_id)
                if self.employee_info: # Check if info was retrieved
                     logging.info(f"Stable identification: Tracker {self.track_id} identified as Employee {best_match_id} ({self.employee_info.get('name','?')})")
                     emp_db.update_employee(best_match_id, 'last_detected', time.strftime("%Y-%m-%d %H:%M:%S"))
                     log_employee_activity(best_match_id, "detected")
                else:
                     logging.warning(f"Stable match for Tracker {self.track_id} to Emp {best_match_id}, but failed to retrieve info.")
                     self.stable = False # Revert stability if info missing

        else: # No match found above the threshold
            self.reset_match_state()

        # Return the current best match ID (even if not stable), actual confidence, and info
        return self.last_matched_id, self.match_confidence, self.employee_info


    def reset_match_state(self):
        """Reset matching variables."""
        self.last_matched_id = None
        self.match_confidence = 0.0
        self.consecutive_matches = 0
        self.stable = False
        self.employee_info = None


# ---------------------------------------
# Employee Database Class (Unchanged)
# ---------------------------------------
# ... (Keep the full EmployeeDatabase class from the previous version here) ...
class EmployeeDatabase:
    def __init__(self, db_file=EMP_DB_FILE):
        self.db_file = db_file
        self.employees = {} # Store as {emp_id (int): data_dict}
        self.load()

    def load(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r") as f:
                    data = json.load(f)
                self.employees = {}
                for emp_id_str, emp_data in data.items():
                    try:
                         emp_id = int(emp_id_str)
                         if 'enhanced_feature' in emp_data and emp_data['enhanced_feature'] is not None:
                              feature_list = emp_data['enhanced_feature']
                              if isinstance(feature_list, list) and len(feature_list) == FEATURE_DIMENSION:
                                   emp_data['enhanced_feature'] = np.array(feature_list, dtype=np.float32)
                              else:
                                   logging.warning(f"Invalid feature data for Emp {emp_id}, removing feature.")
                                   emp_data.pop('enhanced_feature', None)
                         self.employees[emp_id] = emp_data
                    except ValueError:
                         logging.error(f"Invalid employee ID '{emp_id_str}' found in DB file. Skipping.")
                logging.info(f"Loaded {len(self.employees)} employees from {self.db_file}")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from {self.db_file}: {e}")
                self.employees = {}
            except Exception as e:
                logging.error(f"Failed to load employee database: {e}")
                self.employees = {}
        else:
            logging.info("Employee database file not found. Starting with empty database.")
            self.employees = {}

    def save(self):
        save_data = {}
        for emp_id, emp_data in self.employees.items():
            data_to_save = emp_data.copy()
            if 'enhanced_feature' in data_to_save and isinstance(data_to_save['enhanced_feature'], np.ndarray):
                data_to_save['enhanced_feature'] = data_to_save['enhanced_feature'].tolist()
            save_data[str(emp_id)] = data_to_save

        try:
            with open(self.db_file, "w") as f:
                json.dump(save_data, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save employee database: {e}")

    def add_employee(self, emp_id, name, department, enhanced_feature=None):
        if not isinstance(emp_id, int):
            logging.error("Employee ID must be an integer.")
            return False
        if emp_id in self.employees:
             logging.warning(f"Employee ID {emp_id} already exists. Updating info.")

        self.employees[emp_id] = {
            'name': name, 'department': department,
            'registered_date': self.employees.get(emp_id, {}).get('registered_date', time.strftime("%Y-%m-%d")),
            'last_detected': self.employees.get(emp_id, {}).get('last_detected', None)
        }

        if enhanced_feature is not None:
            if isinstance(enhanced_feature, np.ndarray) and enhanced_feature.shape == (FEATURE_DIMENSION,):
                 self.employees[emp_id]['enhanced_feature'] = enhanced_feature.astype(np.float32)
            else:
                 logging.error(f"Invalid feature vector provided for Emp {emp_id}. Feature not added.")

        self.save()
        logging.info(f"Employee {emp_id} added/updated.")
        return True

    def update_employee(self, emp_id, field, value):
        if emp_id in self.employees:
            if field == 'enhanced_feature':
                 if isinstance(value, np.ndarray) and value.shape == (FEATURE_DIMENSION,):
                      self.employees[emp_id][field] = value.astype(np.float32)
                      log_employee_activity(emp_id, "features_registered")
                 else:
                      logging.error(f"Invalid feature vector provided for update on Emp {emp_id}.")
                      return False
            else:
                 self.employees[emp_id][field] = value
            self.save()
            return True
        logging.warning(f"Attempted to update non-existent employee {emp_id}.")
        return False

    def delete_employee(self, emp_id):
        if emp_id in self.employees:
            del self.employees[emp_id]
            self.save()
            logging.info(f"Employee {emp_id} deleted.")
            return True
        logging.warning(f"Attempted to delete non-existent employee {emp_id}.")
        return False

    def get_employee_info(self, emp_id):
        return self.employees.get(emp_id, None)

    def get_all_employees(self):
        return self.employees

    def has_enhanced_feature(self, emp_id):
        return emp_id in self.employees and \
               'enhanced_feature' in self.employees[emp_id] and \
               isinstance(self.employees[emp_id]['enhanced_feature'], np.ndarray)


# -------------------------------
# Visualization Function (Same as before)
# -------------------------------
# ... (Keep the full visualize_features_internal function from the previous version here) ...
def visualize_features_internal(tracker_data, employee_db, parent_window):
    """Internal function to create feature visualization plot."""
    viz_window = tk.Toplevel(parent_window)
    viz_window.title("Feature Visualization")
    viz_window.geometry("900x700")

    fig = plt.Figure(figsize=(10, 7), dpi=100)
    ax1 = fig.add_subplot(211) # Comparison plot
    ax2 = fig.add_subplot(212) # Dynamics plot

    stable_tracker = None
    for tracker in tracker_data.values():
        if tracker.stable and tracker.last_matched_id is not None:
            stable_tracker = tracker
            break

    if stable_tracker:
        emp_id = stable_tracker.last_matched_id
        emp_data = employee_db.get_employee_info(emp_id)

        current_feature = stable_tracker.get_identity_feature()
        template_feature = emp_data.get('enhanced_feature') if emp_data else None

        if current_feature is not None and template_feature is not None and isinstance(template_feature, np.ndarray):
            if current_feature.shape == template_feature.shape:
                x = np.arange(len(current_feature))
                ax1.bar(x - 0.2, current_feature, width=0.4, label="Current Track", color='blue', alpha=0.7)
                ax1.bar(x + 0.2, template_feature, width=0.4, label="DB Template", color='green', alpha=0.7)
                ax1.set_title(f"Feature Vector Comparison: Emp {emp_id} ({emp_data.get('name', 'N/A')})")
                ax1.set_xlabel("Feature Index")
                ax1.set_ylabel("Value")
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.5)

                similarity = stable_tracker.feature_extractor.match_features(current_feature, template_feature, threshold=0.0) # Get raw similarity
                ax1.text(0.95, 0.95, f"Similarity: {similarity:.3f}", transform=ax1.transAxes,
                         fontsize=10, verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                 ax1.text(0.5, 0.5, "Feature shape mismatch", ha='center', va='center', fontsize=12, color='red')
        else:
            ax1.text(0.5, 0.5, "Feature data unavailable", ha='center', va='center', fontsize=12)

        if len(stable_tracker.feature_buffer) >= 5:
            features_over_time = np.array(stable_tracker.feature_buffer)
            time_steps = np.arange(len(features_over_time))
            num_features_to_plot = min(4, features_over_time.shape[1])
            for i in range(num_features_to_plot):
                ax2.plot(time_steps, features_over_time[:, i], label=f"Raw Feature {i+1}", alpha=0.8)
            ax2.set_title(f"Raw Feature Dynamics (Last {len(time_steps)} Frames)")
            ax2.set_xlabel("Frame Index in Buffer")
            ax2.set_ylabel("Value")
            ax2.legend(fontsize='small')
            ax2.grid(True, linestyle='--', alpha=0.5)
        else:
             ax2.text(0.5, 0.5, "Insufficient data for dynamics plot", ha='center', va='center', fontsize=12)

    else:
        ax1.text(0.5, 0.5, "No stable identified person found for comparison", ha='center', va='center', fontsize=12)
        ax2.text(0.5, 0.5, "No stable tracker data", ha='center', va='center', fontsize=12)

    fig.tight_layout(pad=3.0)
    canvas = FigureCanvasTkAgg(fig, master=viz_window)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    close_button = ttk.Button(viz_window, text="Close", command=viz_window.destroy)
    close_button.pack(pady=10)


# -------------------------------
# Main Application Class (Modified for Multi-Person)
# -------------------------------
class WarehouseTrackingApp:
    def __init__(self, root, video_source=0):
        self.root = root
        self.root.title("Multi-Person Warehouse Employee Tracking System")
        self.root.geometry("1300x750")
        self.video_source = video_source

        self.is_tracking = False
        self.capture_thread = None
        self.yolo_model = None # Placeholder for YOLO model
        self.pose_detector = PoseDetector()
        self.emp_db = EmployeeDatabase()
        self.trackers = {}
        self.next_track_id = 0

        self.registration_mode = False
        self.registering_employee = None

        # --- Load YOLO Model ---
        try:
            logging.info(f"Loading YOLO model ({YOLO_MODEL_PATH}) onto device: {DEVICE}")
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            self.yolo_model.to(DEVICE) # Move model to GPU if available
            logging.info("YOLO model loaded successfully.")
        except Exception as e:
            logging.critical(f"Failed to load YOLO model: {e}", exc_info=True)
            messagebox.showerror("Model Error", f"Failed to load YOLO model: {e}\nApplication cannot start tracking.")
            # Optionally disable tracking button or exit
            # self.root.destroy() # Exit if YOLO is critical
            # For now, allow GUI to load but tracking won't work well

        self.setup_ui()
        self.refresh_employee_list()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        logging.info("GUI Initialized.")

    def setup_ui(self): # Largely the same as before
        # ... (Keep the setup_ui method from the previous version) ...
         # Main Frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Frame (Video)
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.canvas = tk.Canvas(self.left_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right Frame (Controls & Employee List)
        self.right_frame = ttk.Frame(self.main_frame, width=400) # Fixed width
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        self.right_frame.pack_propagate(False)

        # --- Control Panel ---
        self.control_frame = ttk.LabelFrame(self.right_frame, text="Controls", padding="10")
        self.control_frame.pack(fill=tk.X, pady=(0, 10))

        self.track_btn = ttk.Button(self.control_frame, text="Start Tracking", command=self.toggle_tracking)
        self.track_btn.pack(fill=tk.X, pady=5)
        # Disable button if YOLO failed to load
        if self.yolo_model is None:
            self.track_btn.config(state=tk.DISABLED)


        self.visualize_btn = ttk.Button(self.control_frame, text="Visualize Features", command=self.visualize_features)
        self.visualize_btn.pack(fill=tk.X, pady=5)

        # --- Employee Management ---
        self.emp_frame = ttk.LabelFrame(self.right_frame, text="Employee Management", padding="10")
        self.emp_frame.pack(fill=tk.BOTH, expand=True)

        # Treeview
        self.tree_columns = ("ID", "Name", "Department", "Registered", "Last Seen")
        self.tree = ttk.Treeview(self.emp_frame, columns=self.tree_columns, show="headings", height=15)

        # Columns Config
        self.tree.heading("ID", text="ID", anchor=tk.W); self.tree.column("ID", width=50, stretch=False, anchor=tk.W)
        self.tree.heading("Name", text="Name", anchor=tk.W); self.tree.column("Name", width=120, stretch=True, anchor=tk.W)
        self.tree.heading("Department", text="Dept", anchor=tk.W); self.tree.column("Department", width=80, stretch=False, anchor=tk.W)
        self.tree.heading("Registered", text="Reg.", anchor=tk.CENTER); self.tree.column("Registered", width=40, stretch=False, anchor=tk.CENTER)
        self.tree.heading("Last Seen", text="Last Seen", anchor=tk.W); self.tree.column("Last Seen", width=130, stretch=False, anchor=tk.W)

        # Scrollbars
        vsb = ttk.Scrollbar(self.emp_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.emp_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Tags Config
        self.tree.tag_configure('registered', foreground='green')
        self.tree.tag_configure('unregistered', foreground='red')

        # Action Buttons Frame
        action_btn_frame = ttk.Frame(self.emp_frame)
        action_btn_frame.pack(fill=tk.X, pady=5)
        self.add_btn = ttk.Button(action_btn_frame, text="Add", command=self.show_add_employee_dialog)
        self.add_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        self.register_btn = ttk.Button(action_btn_frame, text="Register Features", command=self.initiate_registration)
        self.register_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        self.delete_btn = ttk.Button(action_btn_frame, text="Delete", command=self.delete_selected_employee)
        self.delete_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("System Ready.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding="2 5")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)


    # --- Other UI Methods (refresh_employee_list, show_add_employee_dialog, etc.) ---
    # ... (Keep these methods largely the same as the previous version) ...
    def refresh_employee_list(self): # Same
        for item in self.tree.get_children(): self.tree.delete(item)
        sorted_emp_ids = sorted(self.emp_db.get_all_employees().keys())
        for emp_id in sorted_emp_ids:
             emp_data = self.emp_db.get_employee_info(emp_id)
             if emp_data:
                 registered_status = "✓" if self.emp_db.has_enhanced_feature(emp_id) else "✗"
                 tag = 'registered' if registered_status == "✓" else 'unregistered'
                 last_seen = emp_data.get('last_detected', 'Never')
                 values = (emp_id, emp_data.get('name', 'N/A'), emp_data.get('department', 'N/A'), registered_status, last_seen)
                 self.tree.insert('', 'end', values=values, tags=(tag,))

    def show_add_employee_dialog(self): # Same
        dialog = tk.Toplevel(self.root); dialog.title("Add New Employee"); dialog.geometry("300x200")
        dialog.resizable(False, False); dialog.grab_set()
        frame = ttk.Frame(dialog, padding="10"); frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Employee ID:").grid(row=0, column=0, sticky=tk.W, pady=5)
        id_var = tk.StringVar(); ttk.Entry(frame, textvariable=id_var).grid(row=0, column=1, sticky=tk.EW, pady=5)
        ttk.Label(frame, text="Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        name_var = tk.StringVar(); ttk.Entry(frame, textvariable=name_var).grid(row=1, column=1, sticky=tk.EW, pady=5)
        ttk.Label(frame, text="Department:").grid(row=2, column=0, sticky=tk.W, pady=5)
        dept_var = tk.StringVar(); ttk.Entry(frame, textvariable=dept_var).grid(row=2, column=1, sticky=tk.EW, pady=5)
        frame.columnconfigure(1, weight=1)
        def save():
            try:
                emp_id = int(id_var.get().strip()); name = name_var.get().strip(); dept = dept_var.get().strip()
                if not emp_id or not name: messagebox.showerror("Input Error", "ID and Name required.", parent=dialog); return
                if emp_id in self.emp_db.employees: messagebox.showwarning("Duplicate ID", f"ID {emp_id} exists. Info will be updated.", parent=dialog)
                if self.emp_db.add_employee(emp_id, name, dept):
                    self.refresh_employee_list(); dialog.destroy(); messagebox.showinfo("Success", f"Employee {emp_id} added/updated.", parent=self.root)
                else: messagebox.showerror("Error", "Failed to save employee.", parent=dialog)
            except ValueError: messagebox.showerror("Input Error", "ID must be integer.", parent=dialog)
            except Exception as e: messagebox.showerror("Error", f"An error occurred: {e}", parent=dialog); logging.error(f"Error adding employee: {e}")
        button_frame = ttk.Frame(frame); button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(button_frame, text="Save", command=save).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        dialog.wait_window()

    def initiate_registration(self): # Same
        selected_item = self.tree.focus()
        if not selected_item: messagebox.showwarning("Selection Error", "Select employee first.", parent=self.root); return
        if not self.is_tracking: messagebox.showinfo("Start Tracking", "Tracking must be active.", parent=self.root); return
        item_values = self.tree.item(selected_item, 'values')
        try: emp_id = int(item_values[0])
        except (ValueError, IndexError): messagebox.showerror("Error", "Could not get employee ID.", parent=self.root); return
        self.registration_mode = True; self.registering_employee = emp_id
        self.status_var.set(f"REGISTRATION ACTIVE: Ask Employee {emp_id} to walk in center.")
        logging.info(f"Initiating registration for employee {emp_id}.")
        messagebox.showinfo("Registration Mode", f"Reg. active for {emp_id}.\nWalk naturally in view.", parent=self.root)

    def delete_selected_employee(self): # Same
        selected_item = self.tree.focus()
        if not selected_item: messagebox.showwarning("Selection Error", "Select employee to delete.", parent=self.root); return
        item_values = self.tree.item(selected_item, 'values')
        try: emp_id = int(item_values[0]); emp_name = item_values[1]
        except (ValueError, IndexError): messagebox.showerror("Error", "Could not get employee ID.", parent=self.root); return
        if messagebox.askyesno("Confirm Deletion", f"Delete {emp_id} ({emp_name})?", parent=self.root):
            if self.emp_db.delete_employee(emp_id):
                self.refresh_employee_list(); messagebox.showinfo("Success", f"Employee {emp_id} deleted.", parent=self.root)
            else: messagebox.showerror("Error", f"Failed to delete {emp_id}.", parent=self.root)


    def toggle_tracking(self): # Same logic, but check yolo_model exists
        if self.yolo_model is None:
             messagebox.showerror("Model Error", "YOLO model not loaded. Cannot start tracking.")
             return

        if self.is_tracking:
            self.is_tracking = False
            self.track_btn.config(text="Start Tracking")
            self.status_var.set("Tracking stopped.")
            logging.info("Tracking stopped by user.")
            self.trackers = {}
            self.next_track_id = 0
            if self.capture_thread and self.capture_thread.is_alive():
                 self.capture_thread.join(timeout=0.5)
        else:
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                messagebox.showerror("Camera Error", f"Could not open video source {self.video_source}.")
                logging.error(f"Failed to open video source {self.video_source}.")
                cap.release()
                return
            cap.release()

            self.is_tracking = True
            self.track_btn.config(text="Stop Tracking")
            self.status_var.set("Tracking active...")
            logging.info("Tracking started.")
            self.trackers = {}
            self.next_track_id = 0
            self.capture_thread = Thread(target=self.tracking_loop, daemon=True)
            self.capture_thread.start()

    # --- MODIFIED Tracking Loop ---
    def tracking_loop(self):
        """Main loop for multi-person detection, pose estimation, and tracking."""
        if self.yolo_model is None:
             logging.error("Tracking loop started but YOLO model is not loaded.")
             self.is_tracking = False
             self.root.after(0, lambda: self.status_var.set("Error: YOLO model failed."))
             self.root.after(0, lambda: self.track_btn.config(text="Start Tracking", state=tk.DISABLED))
             return

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            logging.error(f"Camera {self.video_source} failed in tracking loop.")
            self.root.after(0, lambda: self.status_var.set("Error: Camera disconnected."))
            self.root.after(0, lambda: self.track_btn.config(text="Start Tracking"))
            self.is_tracking = False
            return

        fps_time = time.time()
        frame_count = 0

        while self.is_tracking:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab frame.")
                time.sleep(0.1)
                continue

            frame_h, frame_w = frame.shape[:2]
            
            # --- Stage 1: YOLO Person Detection ---
            try:
                # Use half precision (FP16) if on GPU for speed, FP32 on CPU
                # verbose=False reduces console output
                yolo_results = self.yolo_model(frame, device=DEVICE, verbose=False, half=USE_GPU)

            except Exception as e:
                 logging.error(f"Error during YOLO detection: {e}")
                 yolo_results = [] # Handle error

            # --- Stage 2: Process Detections & Get Poses ---
            current_detections = [] # Stores {'bbox': (x1,y1,x2,y2), 'landmarks': translated_landmarks_list}
            if yolo_results: # Check if YOLO returned results
                 # Process results from the first image (index 0)
                 boxes = yolo_results[0].boxes
                 for box in boxes:
                     # Check confidence and class
                     conf = box.conf[0].item() # Get confidence score as float
                     cls_id = int(box.cls[0].item()) # Get class ID as int

                     if cls_id == PERSON_CLASS_ID and conf >= PERSON_CONF_THRESHOLD:
                         # Get bounding box coordinates (xyxy format)
                         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                         # Clamp coordinates to frame boundaries
                         x1, y1 = max(0, x1), max(0, y1)
                         x2, y2 = min(frame_w, x2), min(frame_h, y2)

                         # --- Crop Frame ---
                         if x1 >= x2 or y1 >= y2: continue # Skip invalid boxes
                         crop = frame[y1:y2, x1:x2]
                         if crop.size == 0: continue # Skip empty crops

                         # --- Run Pose Estimation on Crop ---
                         try:
                            pose_landmarks_on_crop = self.pose_detector.process_frame(crop)
                         except Exception as e:
                             logging.error(f"Error processing pose on crop: {e}")
                             pose_landmarks_on_crop = None


                         # --- Translate Landmarks (if pose found) ---
                         translated_landmarks_list = None
                         if pose_landmarks_on_crop and pose_landmarks_on_crop.landmark:
                             crop_h, crop_w = crop.shape[:2]
                             translated_landmarks_list = []
                             for lm in pose_landmarks_on_crop.landmark:
                                 # Create a new landmark object to avoid modifying the original results
                                 new_lm = mp.solutions.pose.PoseLandmark(value=0)._replace( # Hacky way to create one
                                     x = (lm.x * crop_w + x1) / frame_w, # Translate X and normalize to full frame
                                     y = (lm.y * crop_h + y1) / frame_h, # Translate Y and normalize to full frame
                                     z = lm.z, # Z is relative, keep as is
                                     visibility = lm.visibility
                                 )
                                 translated_landmarks_list.append(new_lm)

                             # Store detection bbox and translated, normalized landmarks list
                             detection_bbox = (x1, y1, x2, y2) # Use original detection bbox
                             current_detections.append({
                                 'bbox': detection_bbox,
                                 'landmarks': translated_landmarks_list
                             })
                             # Draw raw landmarks on the full frame for visualization
                             # Need to create a temporary landmarks object MediaPipe understands
                             # This part is tricky - drawing directly needs the object structure
                             # For simplicity, we might skip drawing raw landmarks here or find a way to reconstruct
                             # Let's draw the translated points manually for verification (optional)
                             # for lm in translated_landmarks_list:
                             #      cv2.circle(frame, (int(lm.x * frame_w), int(lm.y * frame_h)), 2, (255,0,0), -1)

                         # Optional: Draw YOLO box even if pose fails
                         # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1) # Draw YOLO box in Magenta

            # --- Stage 3: Tracker Association (Simplified Greedy IoU Matching) ---
            # Calculate IoU between existing trackers and new detections
            num_trackers = len(self.trackers)
            num_detections = len(current_detections)
            iou_matrix = np.zeros((num_trackers, num_detections))

            tracker_ids_list = list(self.trackers.keys()) # Get current tracker IDs

            for i, track_id in enumerate(tracker_ids_list):
                tracker = self.trackers[track_id]
                for j, det in enumerate(current_detections):
                    # Use tracker's last bbox and detection's bbox
                    _, iou = check_overlap(tracker.bbox, det['bbox'])
                    iou_matrix[i, j] = iou

            matched_indices = set() # Indices of detections that are matched
            tracker_matches = {} # Stores {track_id: detection_index}

            # Greedy matching: find best match for each tracker
            # More robust: Use Hungarian algorithm (scipy.optimize.linear_sum_assignment) on cost matrix (1-IoU)
            if num_trackers > 0 and num_detections > 0:
                 # Iterate trackers
                 for i, track_id in enumerate(tracker_ids_list):
                     best_iou = IOU_THRESHOLD
                     best_det_idx = -1
                     # Find best detection for this tracker
                     for j in range(num_detections):
                          if j not in matched_indices and iou_matrix[i, j] > best_iou:
                              best_iou = iou_matrix[i, j]
                              best_det_idx = j

                     # If a match found for this tracker
                     if best_det_idx != -1:
                          tracker_matches[track_id] = best_det_idx
                          matched_indices.add(best_det_idx)


            # --- Stage 4: Update Trackers and Handle Timeouts ---
            for track_id, tracker in list(self.trackers.items()): # Iterate copy
                if track_id in tracker_matches:
                    det_idx = tracker_matches[track_id]
                    detection = current_detections[det_idx]
                    # Update tracker position with the matched detection's bbox
                    # Calculate center for the tracker bbox
                    x1, y1, x2, y2 = detection['bbox']
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    tracker.update_position((x1, y1, x2, y2, cx, cy))
                    # Add the translated landmarks
                    tracker.add_landmarks(detection['landmarks'], (frame_h, frame_w))
                else:
                    # No match found, check for timeout
                    if time.time() - tracker.last_seen > TRACKER_TIMEOUT_SEC:
                        logging.info(f"Tracker {track_id} timed out.")
                        del self.trackers[track_id]

            # --- Stage 5: Create New Trackers for Unmatched Detections ---
            for j, det in enumerate(current_detections):
                if j not in matched_indices:
                    if len(self.trackers) < MAX_PEOPLE_TRACK:
                        # Calculate center for the new tracker bbox
                        x1, y1, x2, y2 = det['bbox']
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        new_tracker = EnhancedPersonTracker(self.next_track_id, (x1, y1, x2, y2, cx, cy))
                        new_tracker.add_landmarks(det['landmarks'], (frame_h, frame_w))
                        self.trackers[self.next_track_id] = new_tracker
                        logging.debug(f"New tracker created: ID {self.next_track_id}")
                        self.next_track_id += 1
                    else:
                        logging.warning("Max trackers reached.")
                        break # Stop adding new trackers

            # --- Stage 6: Process Trackers (Registration & Identification & Drawing) ---
            active_tracker_ids_this_frame = set(self.trackers.keys()) # Keep track of IDs still active

            for track_id in active_tracker_ids_this_frame:
                 # Check if tracker still exists (might have been deleted by timeout logic just before)
                 if track_id not in self.trackers: continue

                 tracker = self.trackers[track_id]
                 x_min, y_min, x_max, y_max, center_x, center_y = tracker.bbox # Use tracker's current bbox
                 box_color = (255, 150, 0) # Default Orange
                 label = f"TrackID: {track_id}"

                 # --- Registration ---
                 if self.registration_mode and self.registering_employee is not None:
                     frame_center_x = frame_w / 2
                     is_centered = abs(center_x - frame_center_x) < (frame_w * REGISTRATION_CENTER_THRESHOLD_FACTOR)
                     if is_centered:
                         box_color = (0, 255, 255) # Yellow
                         label += " (Centering)"
                         identity_feature = tracker.get_identity_feature()
                         if identity_feature is not None:
                             label += " (Ready!)"
                             box_color = (0, 0, 255) # Red
                             if self.emp_db.update_employee(self.registering_employee, 'enhanced_feature', identity_feature):
                                 logging.info(f"Registered features for Emp {self.registering_employee} from Tracker {track_id}.")
                                 self.root.after(0, lambda eid=self.registering_employee: self.status_var.set(f"Features registered for Emp {eid}!"))
                                 self.root.after(0, self.refresh_employee_list)
                                 self.root.after(0, lambda eid=self.registering_employee: messagebox.showinfo("Registration Complete", f"Features registered for Emp {eid}."))
                             else:
                                  logging.error(f"Failed to save features for Emp {self.registering_employee}.")
                                  self.root.after(0, lambda eid=self.registering_employee: self.status_var.set(f"Error saving features for {eid}."))
                             self.registration_mode = False; self.registering_employee = None

                 # --- Identification ---
                 elif not self.registration_mode:
                     emp_id, confidence, emp_info = tracker.match_with_database(self.emp_db)
                     if tracker.stable and emp_id is not None and emp_info is not None:
                         box_color = (0, 255, 0) # Green
                         emp_name = emp_info.get('name', '?')
                         label = f"ID: {emp_id} ({emp_name}) {confidence:.2f}"
                     elif emp_id is not None:
                         box_color = (255, 255, 0) # Cyan
                         label = f"Match: {emp_id}? ({confidence:.2f})"

                 # --- Drawing ---
                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                 label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                 label_ymin = max(y_min, label_size[1] + 5)
                 cv2.rectangle(frame, (x_min, label_ymin - label_size[1] - 5), (x_min + label_size[0], label_ymin - base_line), box_color, cv2.FILLED)
                 cv2.putText(frame, label, (x_min, label_ymin - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


            # --- Display FPS & Registration Banner (Same as before) ---
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                frame_count = 0; fps_time = current_time
            if self.registration_mode:
                 cv2.putText(frame, f"REG MODE: Emp {self.registering_employee}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


            # --- Update Canvas (Same as before) ---
            try:
                 cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                 pil_image = Image.fromarray(cv2image)
                 photo = ImageTk.PhotoImage(image=pil_image)
                 self.root.after(0, self.update_canvas, photo)
            except Exception as e:
                 logging.error(f"Error converting/displaying frame: {e}")


        # --- Cleanup ---
        cap.release()
        self.pose_detector.close()
        logging.info("Video capture released and pose detector closed.")
        self.root.after(0, lambda: self.status_var.set("Tracking stopped."))

    def update_canvas(self, photo): # Same as before
        if self.canvas.winfo_exists():
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo
        else:
            logging.warning("Canvas widget no longer exists, cannot update.")

    def visualize_features(self): # Same as before
        if not self.trackers: messagebox.showinfo("No Data", "No active trackers.", parent=self.root); return
        visualize_features_internal(self.trackers, self.emp_db, self.root)

    def on_close(self): # Same as before
        logging.info("Close request received. Shutting down.")
        if self.is_tracking:
            self.is_tracking = False
            if self.capture_thread and self.capture_thread.is_alive():
                 self.capture_thread.join(timeout=1.0)
        self.emp_db.save()
        logging.info("Employee database saved.")
        # Explicitly close pose detector if not already closed
        if hasattr(self, 'pose_detector') and self.pose_detector:
            self.pose_detector.close()
        self.root.destroy()
        logging.info("Application closed.")


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = WarehouseTrackingApp(root, video_source=0) # Use 0 for default camera
        root.mainloop()
    except Exception as e:
         logging.critical(f"Unhandled exception in main application: {e}", exc_info=True)
         try: messagebox.showerror("Critical Error", f"A critical error occurred:\n{e}\nCheck logs.")
         except: print(f"CRITICAL ERROR: {e}")