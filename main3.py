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
# Optional, but good for feature normalization if needed later
# from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------------------
# Configuration and Settings
# -------------------------------

EMP_DB_FILE = "employee_enhanced_db.json" # Database file
LOG_FILE = "warehouse_tracking.log"      # Log file
ACTIVITY_LOG_FILE = "warehouse_activity_log.csv" # Employee activity log

FEATURE_BUFFER_LENGTH = 45  # Number of frames for feature aggregation (adjust based on typical walk cycle duration)
MIN_DETECTION_FRAMES = 10   # Min consecutive frames for stable ID
MAX_PEOPLE_TRACK = 10       # Max people to track simultaneously
MATCH_THRESHOLD = 0.70      # Cosine similarity threshold for matching (0.0 to 1.0)
FEATURE_DIMENSION = 54      # Expected dimension of the processed feature vector (adjust if extractor changes)
REGISTRATION_CENTER_THRESHOLD_FACTOR = 0.25 # How central (fraction of width) person needs to be for registration
IOU_THRESHOLD = 0.4         # Intersection over Union threshold for tracker association

# -------------------------------
# Logging Setup
# -------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler() # Also print logs to console
    ]
)
logging.info("Application started.")

# -------------------------------
# Helper Functions
# -------------------------------

def compute_angle(a, b, c):
    """Compute the angle (in degrees) between three points: a, b, c with b as the vertex."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    # Handle potential zero vectors
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0 # Or handle as appropriate (e.g., raise error, return NaN)
    
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    # Clip value to handle potential floating point inaccuracies outside [-1, 1]
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_person_bbox(landmarks, frame_shape):
    """Get the bounding box coordinates for a person based on pose landmarks."""
    h, w = frame_shape[:2]
    visible_landmarks = [lm for lm in landmarks if lm.visibility > 0.5]
    if not visible_landmarks:
        # Fallback or error handling if no visible landmarks
        return 0, 0, w, h, w // 2, h // 2 
    
    x_coords = [lm.x * w for lm in visible_landmarks]
    y_coords = [lm.y * h for lm in visible_landmarks]

    padding_factor = 0.1 # Percentage padding
    x_min = max(0, int(min(x_coords) - padding_factor * (max(x_coords) - min(x_coords))))
    y_min = max(0, int(min(y_coords) - padding_factor * (max(y_coords) - min(y_coords))))
    x_max = min(w, int(max(x_coords) + padding_factor * (max(x_coords) - min(x_coords))))
    y_max = min(h, int(max(y_coords) + padding_factor * (max(y_coords) - min(y_coords))))
    
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    return (x_min, y_min, x_max, y_max, center_x, center_y)

def check_overlap(bbox1, bbox2, iou_threshold=IOU_THRESHOLD):
    """Check if two bounding boxes overlap significantly based on IoU."""
    x1_min, y1_min, x1_max, y1_max = bbox1[:4]
    x2_min, y2_min, x2_max, y2_max = bbox2[:4]
    
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return False, 0.0 # No overlap
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = float(box1_area + box2_area - intersection_area) # Use float for division
    
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou > iou_threshold, iou

def log_employee_activity(emp_id, action="detected"):
    """Log employee activity to CSV file."""
    try:
        file_exists = os.path.exists(ACTIVITY_LOG_FILE)
        with open(ACTIVITY_LOG_FILE, 'a') as f:
            if not file_exists:
                f.write("timestamp,employee_id,action\n") # Write header if new file
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp},{emp_id},{action}\n")
    except Exception as e:
        logging.error(f"Failed to log activity for {emp_id}: {e}")

# -------------------------------
# MediaPipe Pose Detector Class
# -------------------------------
class PoseDetector:
    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def process_frame(self, frame):
        """Process a frame to detect pose landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False # Improve performance
        results = self.pose.process(rgb_frame)
        rgb_frame.flags.writeable = True 
        
        # Return only landmarks if detected
        if results.pose_landmarks:
            # Currently returns single most prominent person.
            # For true multi-person, an object detector (like YOLO) + Pose on each detected person is needed.
            # Let's assume for now MediaPipe provides the best single detection or we adapt based on it.
            return [results.pose_landmarks] 
        return []
    
    def draw_landmarks(self, frame, landmarks):
        """Draw pose landmarks on the frame."""
        self.mp_drawing.draw_landmarks(
            frame, 
            landmarks, 
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        return frame
        
    def close(self):
        self.pose.close()

# -------------------------------
# Enhanced Feature Extraction
# -------------------------------
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
            lm = landmarks[landmark_enum.value]
            if lm.visibility < 0.3: # Threshold for considering a landmark visible
                return None
            return np.array([lm.x * w, lm.y * h])
        except IndexError:
            logging.warning(f"Landmark {landmark_enum.name} index out of bounds.")
            return None
        except AttributeError:
            logging.warning(f"Landmark {landmark_enum.name} not found in results.")
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

        for lm_enum in required_landmarks:
            coords[lm_enum] = self._get_landmark(landmarks, lm_enum, frame_shape)

        # Check if all required landmarks are available
        if any(coord is None for coord in coords.values()):
            logging.debug("Missing some required landmarks for angle calculation.")
            # Return a zero vector or handle appropriately
            return np.zeros(8) 

        angles = []
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
        mid_shoulder = (coords[self.mp_pose.PoseLandmark.LEFT_SHOULDER] + coords[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]) / 2
        mid_hip = (coords[self.mp_pose.PoseLandmark.LEFT_HIP] + coords[self.mp_pose.PoseLandmark.RIGHT_HIP]) / 2
        # Approximate neck position relative to shoulders/nose
        neck = mid_shoulder + (coords[self.mp_pose.PoseLandmark.NOSE] - mid_shoulder) * 0.1 
        angles.append(compute_angle(neck, mid_shoulder, mid_hip))

        # Torso twist (relative shoulder-hip horizontal alignment) - simplified
        shoulder_vec = coords[self.mp_pose.PoseLandmark.RIGHT_SHOULDER] - coords[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        hip_vec = coords[self.mp_pose.PoseLandmark.RIGHT_HIP] - coords[self.mp_pose.PoseLandmark.LEFT_HIP]
        # Use only x-component for rough twist/view angle indicator
        angles.append(abs(shoulder_vec[0] - hip_vec[0])) # Difference in horizontal span

        return np.nan_to_num(np.array(angles)) # Replace NaN with 0

    def extract_body_proportions(self, landmarks, frame_shape):
        """Extract body proportion features."""
        coords = {}
        required_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]
        for lm_enum in required_landmarks:
             coords[lm_enum] = self._get_landmark(landmarks, lm_enum, frame_shape)

        if any(coord is None for coord in coords.values()):
             logging.debug("Missing landmarks for proportion calculation.")
             return np.zeros(3)

        # Calculate distances
        shoulder_width = np.linalg.norm(coords[self.mp_pose.PoseLandmark.RIGHT_SHOULDER] - coords[self.mp_pose.PoseLandmark.LEFT_SHOULDER])
        hip_width = np.linalg.norm(coords[self.mp_pose.PoseLandmark.RIGHT_HIP] - coords[self.mp_pose.PoseLandmark.LEFT_HIP])
        right_upper_leg = np.linalg.norm(coords[self.mp_pose.PoseLandmark.RIGHT_HIP] - coords[self.mp_pose.PoseLandmark.RIGHT_KNEE])
        left_upper_leg = np.linalg.norm(coords[self.mp_pose.PoseLandmark.LEFT_HIP] - coords[self.mp_pose.PoseLandmark.LEFT_KNEE])
        right_lower_leg = np.linalg.norm(coords[self.mp_pose.PoseLandmark.RIGHT_KNEE] - coords[self.mp_pose.PoseLandmark.RIGHT_ANKLE])
        left_lower_leg = np.linalg.norm(coords[self.mp_pose.PoseLandmark.LEFT_KNEE] - coords[self.mp_pose.PoseLandmark.LEFT_ANKLE])
        mid_shoulder = (coords[self.mp_pose.PoseLandmark.LEFT_SHOULDER] + coords[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]) / 2
        mid_hip = (coords[self.mp_pose.PoseLandmark.LEFT_HIP] + coords[self.mp_pose.PoseLandmark.RIGHT_HIP]) / 2
        torso_length = np.linalg.norm(mid_shoulder - mid_hip)

        # Calculate ratios (add epsilon to prevent division by zero)
        epsilon = 1e-6
        shoulder_hip_ratio = shoulder_width / (hip_width + epsilon)
        avg_leg_length = (right_upper_leg + left_upper_leg + right_lower_leg + left_lower_leg) / 2
        leg_torso_ratio = avg_leg_length / (torso_length + epsilon)
        avg_upper_leg = (right_upper_leg + left_upper_leg) / 2
        avg_lower_leg = (right_lower_leg + left_lower_leg) / 2
        upper_lower_leg_ratio = avg_upper_leg / (avg_lower_leg + epsilon)
        
        proportions = np.array([shoulder_hip_ratio, leg_torso_ratio, upper_lower_leg_ratio])
        return np.nan_to_num(proportions)

    def extract_movement_dynamics(self, pose_history):
        """Extract dynamic movement features from pose history."""
        history_len = len(pose_history)
        if history_len < 10:  # Need sufficient history
            return np.zeros(7) # Return default vector

        # Use ankle positions for stride analysis
        # Assuming pose_history stores full landmark objects
        try:
            left_ankle_y = np.array([pose[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y for pose in pose_history])
            right_ankle_y = np.array([pose[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y for pose in pose_history])
            # Average ankle height might be more stable
            avg_ankle_y = (left_ankle_y + right_ankle_y) / 2.0

            # Velocity and Acceleration (use smoothed diff)
            velocity = np.gradient(avg_ankle_y)
            acceleration = np.gradient(velocity)
            
            # Stride Frequency (using FFT)
            fft_result = np.abs(fft.fft(avg_ankle_y - np.mean(avg_ankle_y))) # Remove DC component
            freqs = fft.fftfreq(history_len)
            # Find dominant frequency in typical walking range (e.g., 0.5 Hz to 3 Hz)
            # Assuming video FPS is ~30, this corresponds to index range
            # Needs FPS info for accurate Hz calculation, use dominant peak index for now
            valid_range = (freqs > 0.01) & (freqs < 0.3) # Heuristic range based on frame indices
            dominant_freq_idx = np.argmax(fft_result[valid_range]) if np.any(valid_range) else 0
            stride_frequency = freqs[valid_range][dominant_freq_idx] if dominant_freq_idx > 0 else 0.0

            # Smoothness (Inverse of Jerk magnitude) - Simplified
            jerk = np.gradient(acceleration)
            smoothness = 1.0 / (np.mean(np.abs(jerk)) + 1e-6) # Higher value = smoother

            # Movement Entropy (predictability)
            # Discretize velocity to calculate entropy
            hist, _ = np.histogram(velocity, bins=10, density=True)
            movement_entropy = entropy(hist)

            dynamics = np.array([
                np.mean(np.abs(velocity)), np.std(velocity),         # Avg and Std Velocity
                np.mean(np.abs(acceleration)), np.std(acceleration), # Avg and Std Acceleration
                stride_frequency,                                    # Dominant Stride Freq (approx)
                smoothness,                                          # Movement Smoothness
                movement_entropy                                     # Movement Entropy
            ])
        except (IndexError, AttributeError, ValueError) as e:
             logging.warning(f"Error extracting dynamics: {e}. Returning zeros.")
             dynamics = np.zeros(7)

        return np.nan_to_num(dynamics)

    def extract_combined_features(self, landmarks, frame_shape, pose_history=None):
        """Extract and combine all features."""
        joint_angles = self.extract_joint_angles(landmarks, frame_shape) # 8 features
        body_proportions = self.extract_body_proportions(landmarks, frame_shape) # 3 features
        
        if pose_history is not None:
            dynamics = self.extract_movement_dynamics(pose_history) # 7 features
        else:
            dynamics = np.zeros(7)
            
        # Combine (8 + 3 + 7 = 18 features per frame)
        combined = np.concatenate([joint_angles, body_proportions, dynamics])
        return np.nan_to_num(combined)

    def process_feature_buffer(self, feature_buffer):
        """Process a buffer of per-frame features into a single identity vector."""
        if len(feature_buffer) < FEATURE_BUFFER_LENGTH // 2:
            return None # Not enough data

        features = np.array(feature_buffer) # Shape: (buffer_len, num_features_per_frame)
        
        # Check for NaN/Inf and handle
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
             logging.warning("NaN or Inf detected in feature buffer, replacing with 0.")
             features = np.nan_to_num(features)

        # Calculate statistics across time
        mean_features = np.mean(features, axis=0)
        std_features = np.std(features, axis=0)
        median_features = np.median(features, axis=0)
        # Min/Max or range could also be useful
        # min_features = np.min(features, axis=0)
        # max_features = np.max(features, axis=0)

        # Combine statistical features (mean, std, median)
        # Shape: (3 * num_features_per_frame,)
        processed = np.concatenate([mean_features, std_features, median_features]) 

        # Ensure consistent dimensionality (should match FEATURE_DIMENSION)
        current_dim = len(processed)
        if current_dim != FEATURE_DIMENSION:
             logging.warning(f"Processed feature dimension ({current_dim}) mismatch expected ({FEATURE_DIMENSION}). Adjusting.")
             if current_dim > FEATURE_DIMENSION:
                 processed = processed[:FEATURE_DIMENSION] # Truncate
             else:
                 padding = np.zeros(FEATURE_DIMENSION - current_dim) # Pad
                 processed = np.concatenate([processed, padding])

        # Optional: Normalize the final feature vector
        # if self.scaler_fitted:
        #     processed = self.scaler.transform(processed.reshape(1, -1)).flatten()
        # else:
        #     logging.warning("Scaler not fitted, returning unnormalized features.")
            
        return processed

    def match_features(self, feature_vector, template_feature, threshold=MATCH_THRESHOLD):
        """Match features using cosine similarity."""
        feature_norm = np.linalg.norm(feature_vector)
        template_norm = np.linalg.norm(template_feature)
        
        if feature_norm == 0 or template_norm == 0:
            return 0.0 # Cannot compare zero vectors

        # Cosine Similarity
        similarity = np.dot(feature_vector, template_feature) / (feature_norm * template_norm)
        similarity = np.clip(similarity, 0.0, 1.0) # Ensure value is in [0, 1]

        return similarity if similarity >= threshold else 0.0

# -------------------------------
# Enhanced Person Tracker
# -------------------------------
class EnhancedPersonTracker:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox # Current bounding box (x_min, y_min, x_max, y_max, cx, cy)
        # Stores per-frame combined features
        self.feature_buffer = deque(maxlen=FEATURE_BUFFER_LENGTH) 
         # Stores raw landmark objects or arrays for dynamic analysis
        self.pose_history = deque(maxlen=FEATURE_BUFFER_LENGTH)
        self.last_matched_id = None # Employee ID if identified
        self.match_confidence = 0.0 # Similarity score of the last match
        self.consecutive_matches = 0 # Frames matched with the same ID
        self.last_seen = time.time() # Timestamp of the last update
        self.stable = False # Becomes True after MIN_DETECTION_FRAMES consecutive matches
        self.employee_info = None # Stores {'name': ..., 'department': ...} if identified
        self.feature_extractor = EnhancedFeatureExtractor() # Each tracker has its extractor instance
        self.processed_feature = None # Store the latest aggregated feature vector

    def update_position(self, bbox):
        self.bbox = bbox
        self.last_seen = time.time()
    
    def add_landmarks(self, landmarks, frame_shape):
        """Add pose landmarks, extract features, update history."""
        # Store raw landmarks (use the .landmark attribute from results)
        self.pose_history.append(landmarks.landmark) # Store the list/iterable of landmarks
        
        # Extract features for this frame
        # Pass the .landmark part, not the results object itself
        current_features = self.feature_extractor.extract_combined_features(
            landmarks.landmark, frame_shape, 
            self.pose_history if len(self.pose_history) > 5 else None
        )
        self.feature_buffer.append(current_features)

        # Update the processed feature if buffer is sufficiently full
        if len(self.feature_buffer) >= FEATURE_BUFFER_LENGTH // 2:
             self.processed_feature = self.feature_extractor.process_feature_buffer(self.feature_buffer)


    def get_identity_feature(self):
        """Return the latest processed identity feature vector."""
        return self.processed_feature # Returns None if not enough data yet
        
    def match_with_database(self, emp_db):
        """Match current feature vector with employee database."""
        identity_feature = self.get_identity_feature()
        
        if identity_feature is None:
            # Not enough data to form a stable feature vector yet
            self.reset_match_state()
            return None, 0.0, None

        best_match_id = None
        best_confidence = 0.0
        
        for emp_id, emp_data in emp_db.employees.items():
            # Check if the employee has a registered enhanced feature
            if 'enhanced_feature' in emp_data and isinstance(emp_data['enhanced_feature'], np.ndarray):
                template = emp_data['enhanced_feature']
                # Compare shapes before matching
                if template.shape == identity_feature.shape:
                    similarity = self.feature_extractor.match_features(
                        identity_feature, template # Pass the threshold from config
                    )
                    
                    if similarity > best_confidence:
                        best_confidence = similarity
                        best_match_id = emp_id
                else:
                    logging.warning(f"Shape mismatch for Emp {emp_id}: Template {template.shape}, Current {identity_feature.shape}")

        # Update tracker state based on match result
        if best_match_id is not None: # Found a match above threshold
            if best_match_id == self.last_matched_id:
                self.consecutive_matches += 1
            else: # New match or first match
                self.last_matched_id = best_match_id
                self.match_confidence = best_confidence
                self.consecutive_matches = 1
                self.stable = False # Reset stability on ID change
                self.employee_info = None # Reset info

            # Check for stability
            if not self.stable and self.consecutive_matches >= MIN_DETECTION_FRAMES:
                self.stable = True
                self.employee_info = emp_db.get_employee_info(best_match_id)
                logging.info(f"Stable identification: Tracker {self.track_id} identified as Employee {best_match_id}")
                # Update last detected time in DB and log activity
                emp_db.update_employee(best_match_id, 'last_detected', time.strftime("%Y-%m-%d %H:%M:%S"))
                log_employee_activity(best_match_id, "detected")
        
        else: # No match found above threshold
            self.reset_match_state()
            
        return self.last_matched_id, self.match_confidence, self.employee_info

    def reset_match_state(self):
        """Reset matching variables when no match is found or feature is unstable."""
        self.last_matched_id = None
        self.match_confidence = 0.0
        self.consecutive_matches = 0
        self.stable = False
        self.employee_info = None

# -------------------------------
# Employee Database Class
# -------------------------------
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
                # Convert keys to int and features back to numpy arrays
                self.employees = {}
                for emp_id_str, emp_data in data.items():
                    try:
                         emp_id = int(emp_id_str)
                         if 'enhanced_feature' in emp_data and emp_data['enhanced_feature'] is not None:
                              feature_list = emp_data['enhanced_feature']
                              # Basic validation
                              if isinstance(feature_list, list) and len(feature_list) == FEATURE_DIMENSION:
                                   emp_data['enhanced_feature'] = np.array(feature_list, dtype=np.float32)
                              else:
                                   logging.warning(f"Invalid feature data for Emp {emp_id}, removing feature.")
                                   emp_data.pop('enhanced_feature', None) # Remove invalid feature
                         self.employees[emp_id] = emp_data
                    except ValueError:
                         logging.error(f"Invalid employee ID '{emp_id_str}' found in DB file. Skipping.")
                logging.info(f"Loaded {len(self.employees)} employees from {self.db_file}")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from {self.db_file}: {e}")
                self.employees = {} # Start fresh if file is corrupt
            except Exception as e:
                logging.error(f"Failed to load employee database: {e}")
                self.employees = {}
        else:
            logging.info("Employee database file not found. Starting with empty database.")
            self.employees = {}
    
    def save(self):
        save_data = {}
        for emp_id, emp_data in self.employees.items():
            # Create a copy to modify for saving
            data_to_save = emp_data.copy()
            # Convert numpy array to list for JSON
            if 'enhanced_feature' in data_to_save and isinstance(data_to_save['enhanced_feature'], np.ndarray):
                data_to_save['enhanced_feature'] = data_to_save['enhanced_feature'].tolist()
            save_data[str(emp_id)] = data_to_save # Use string keys for JSON
        
        try:
            with open(self.db_file, "w") as f:
                json.dump(save_data, f, indent=4)
            # logging.debug(f"Employee database saved to {self.db_file}") # Use debug level for frequent saves
        except Exception as e:
            logging.error(f"Failed to save employee database: {e}")
    
    def add_employee(self, emp_id, name, department, enhanced_feature=None):
        """Add or update an employee."""
        if not isinstance(emp_id, int):
            logging.error("Employee ID must be an integer.")
            return False
        if emp_id in self.employees:
             logging.warning(f"Employee ID {emp_id} already exists. Updating info.")
        
        self.employees[emp_id] = {
            'name': name,
            'department': department,
            'registered_date': self.employees.get(emp_id, {}).get('registered_date', time.strftime("%Y-%m-%d")), # Keep original date if exists
            'last_detected': self.employees.get(emp_id, {}).get('last_detected', None) # Keep last detected if exists
        }
        
        if enhanced_feature is not None:
            if isinstance(enhanced_feature, np.ndarray) and enhanced_feature.shape == (FEATURE_DIMENSION,):
                 self.employees[emp_id]['enhanced_feature'] = enhanced_feature
            else:
                 logging.error(f"Invalid feature vector provided for Emp {emp_id}. Feature not added.")
        
        self.save()
        logging.info(f"Employee {emp_id} added/updated.")
        return True
    
    def update_employee(self, emp_id, field, value):
        """Update a specific field for an employee."""
        if emp_id in self.employees:
             # Special handling for feature update
            if field == 'enhanced_feature':
                 if isinstance(value, np.ndarray) and value.shape == (FEATURE_DIMENSION,):
                      self.employees[emp_id][field] = value
                      log_employee_activity(emp_id, "features_registered") # Log feature registration
                 else:
                      logging.error(f"Invalid feature vector provided for update on Emp {emp_id}.")
                      return False
            else:
                 self.employees[emp_id][field] = value
            self.save()
            # logging.debug(f"Updated field '{field}' for employee {emp_id}.")
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
        return emp_id in self.employees and 'enhanced_feature' in self.employees[emp_id] and isinstance(self.employees[emp_id]['enhanced_feature'], np.ndarray)

# -------------------------------
# Visualization Function
# -------------------------------

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
            break # Visualize the first stable identified person found

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
                
                # Display similarity on plot
                similarity = stable_tracker.feature_extractor.match_features(current_feature, template_feature, threshold=0.0) # Get raw similarity
                ax1.text(0.95, 0.95, f"Similarity: {similarity:.3f}", transform=ax1.transAxes, 
                         fontsize=10, verticalalignment='top', horizontalalignment='right', 
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                 ax1.text(0.5, 0.5, "Feature shape mismatch", ha='center', va='center', fontsize=12, color='red')
        else:
            ax1.text(0.5, 0.5, "Feature data unavailable", ha='center', va='center', fontsize=12)

        # Plot dynamics (e.g., first few features from the buffer over time)
        if len(stable_tracker.feature_buffer) >= 5:
            features_over_time = np.array(stable_tracker.feature_buffer)
            time_steps = np.arange(len(features_over_time))
            # Plot first 4 features as example
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

    # Add a close button
    close_button = ttk.Button(viz_window, text="Close", command=viz_window.destroy)
    close_button.pack(pady=10)

# -------------------------------
# Main Application Class
# -------------------------------
class WarehouseTrackingApp:
    def __init__(self, root, video_source=0):
        self.root = root
        self.root.title("Enhanced Warehouse Employee Tracking System")
        # Adjust size as needed
        self.root.geometry("1300x750") 
        self.video_source = video_source
        
        self.is_tracking = False
        self.capture_thread = None
        self.pose_detector = PoseDetector()
        self.emp_db = EmployeeDatabase() # Uses enhanced DB file
        self.trackers = {} # Stores {track_id: EnhancedPersonTracker}
        self.next_track_id = 0
        
        self.registration_mode = False
        self.registering_employee = None # Stores employee ID being registered
        
        self.setup_ui()
        self.refresh_employee_list() # Initial population

        # Set up cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        logging.info("GUI Initialized.")

    def setup_ui(self):
        # Main Frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Frame (Video)
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.canvas = tk.Canvas(self.left_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right Frame (Controls & Employee List)
        self.right_frame = ttk.Frame(self.main_frame, width=350) # Fixed width for controls
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        self.right_frame.pack_propagate(False) # Prevent resizing

        # --- Control Panel ---
        self.control_frame = ttk.LabelFrame(self.right_frame, text="Controls", padding="10")
        self.control_frame.pack(fill=tk.X, pady=(0, 10))

        self.track_btn = ttk.Button(self.control_frame, text="Start Tracking", command=self.toggle_tracking)
        self.track_btn.pack(fill=tk.X, pady=5)

        self.visualize_btn = ttk.Button(self.control_frame, text="Visualize Features", command=self.visualize_features)
        self.visualize_btn.pack(fill=tk.X, pady=5)

        # --- Employee Management ---
        self.emp_frame = ttk.LabelFrame(self.right_frame, text="Employee Management", padding="10")
        self.emp_frame.pack(fill=tk.BOTH, expand=True)

        # Treeview for employee list
        self.tree_columns = ("ID", "Name", "Department", "Registered", "Last Seen")
        self.tree = ttk.Treeview(self.emp_frame, columns=self.tree_columns, show="headings", height=15)
        
        # Configure columns
        self.tree.heading("ID", text="ID", anchor=tk.W)
        self.tree.column("ID", width=50, stretch=False, anchor=tk.W)
        self.tree.heading("Name", text="Name", anchor=tk.W)
        self.tree.column("Name", width=120, stretch=True, anchor=tk.W)
        self.tree.heading("Department", text="Dept", anchor=tk.W)
        self.tree.column("Department", width=80, stretch=False, anchor=tk.W)
        self.tree.heading("Registered", text="Reg.", anchor=tk.CENTER)
        self.tree.column("Registered", width=40, stretch=False, anchor=tk.CENTER)
        self.tree.heading("Last Seen", text="Last Seen", anchor=tk.W)
        self.tree.column("Last Seen", width=130, stretch=False, anchor=tk.W)

        # Scrollbars
        vsb = ttk.Scrollbar(self.emp_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.emp_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Tag configurations for registered status
        self.tree.tag_configure('registered', foreground='green')
        self.tree.tag_configure('unregistered', foreground='red')

        # Employee Action Buttons Frame
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
        self.status_var.set("System Ready. Load database or add employees.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding="2 5")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def refresh_employee_list(self):
        """Refresh the employee treeview."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add employees sorted by ID
        sorted_emp_ids = sorted(self.emp_db.get_all_employees().keys())
        
        for emp_id in sorted_emp_ids:
             emp_data = self.emp_db.get_employee_info(emp_id)
             if emp_data:
                 registered_status = "✓" if self.emp_db.has_enhanced_feature(emp_id) else "✗"
                 tag = 'registered' if registered_status == "✓" else 'unregistered'
                 last_seen = emp_data.get('last_detected', 'Never')
                 
                 values = (
                     emp_id,
                     emp_data.get('name', 'N/A'),
                     emp_data.get('department', 'N/A'),
                     registered_status,
                     last_seen
                 )
                 self.tree.insert('', 'end', values=values, tags=(tag,))

    def show_add_employee_dialog(self):
        """Dialog to add a new employee."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New Employee")
        dialog.geometry("300x200")
        dialog.resizable(False, False)
        dialog.grab_set() # Make modal

        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Employee ID:").grid(row=0, column=0, sticky=tk.W, pady=5)
        id_var = tk.StringVar()
        ttk.Entry(frame, textvariable=id_var).grid(row=0, column=1, sticky=tk.EW, pady=5)

        ttk.Label(frame, text="Name:").grid(row=1, column=0, sticky=tk.W, pady=5)
        name_var = tk.StringVar()
        ttk.Entry(frame, textvariable=name_var).grid(row=1, column=1, sticky=tk.EW, pady=5)

        ttk.Label(frame, text="Department:").grid(row=2, column=0, sticky=tk.W, pady=5)
        dept_var = tk.StringVar()
        ttk.Entry(frame, textvariable=dept_var).grid(row=2, column=1, sticky=tk.EW, pady=5)
        
        frame.columnconfigure(1, weight=1) # Make entry fields expand

        def save():
            try:
                emp_id = int(id_var.get().strip())
                name = name_var.get().strip()
                dept = dept_var.get().strip()
                if not emp_id or not name:
                    messagebox.showerror("Input Error", "Employee ID and Name are required.", parent=dialog)
                    return
                if emp_id in self.emp_db.employees:
                     messagebox.showwarning("Duplicate ID", f"Employee ID {emp_id} already exists. Information will be updated if you proceed.", parent=dialog)
                     # Or prevent saving: return
                
                if self.emp_db.add_employee(emp_id, name, dept):
                    self.refresh_employee_list()
                    dialog.destroy()
                    messagebox.showinfo("Success", f"Employee {emp_id} added/updated.", parent=self.root)
                else:
                    messagebox.showerror("Error", "Failed to save employee data.", parent=dialog)
            except ValueError:
                messagebox.showerror("Input Error", "Employee ID must be an integer.", parent=dialog)
            except Exception as e:
                 messagebox.showerror("Error", f"An error occurred: {e}", parent=dialog)
                 logging.error(f"Error adding employee: {e}")

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(button_frame, text="Save", command=save).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        dialog.wait_window() # Wait for dialog to close

    def initiate_registration(self):
        """Start the feature registration process for the selected employee."""
        selected_item = self.tree.focus() # Get selected item ID
        if not selected_item:
            messagebox.showwarning("Selection Error", "Please select an employee from the list first.", parent=self.root)
            return

        if not self.is_tracking:
             messagebox.showinfo("Start Tracking", "Tracking must be active to register features. Please start tracking.", parent=self.root)
             # Optionally, start tracking automatically:
             # self.toggle_tracking()
             # if not self.is_tracking: return # If starting failed
             return
             
        item_values = self.tree.item(selected_item, 'values')
        try:
             emp_id = int(item_values[0])
        except (ValueError, IndexError):
             messagebox.showerror("Error", "Could not get employee ID from selection.", parent=self.root)
             return

        self.registration_mode = True
        self.registering_employee = emp_id
        self.status_var.set(f"REGISTRATION ACTIVE: Ask Employee {emp_id} to walk naturally in the center of the frame.")
        logging.info(f"Initiating feature registration for employee {emp_id}.")
        messagebox.showinfo("Registration Mode", 
                            f"Registration mode active for employee {emp_id}.\n"
                            "Ensure the employee walks naturally within the camera view for ~{FEATURE_BUFFER_LENGTH // 2} frames.", 
                            parent=self.root)

    def delete_selected_employee(self):
        """Delete the employee selected in the treeview."""
        selected_item = self.tree.focus()
        if not selected_item:
            messagebox.showwarning("Selection Error", "Please select an employee to delete.", parent=self.root)
            return
            
        item_values = self.tree.item(selected_item, 'values')
        try:
            emp_id = int(item_values[0])
            emp_name = item_values[1]
        except (ValueError, IndexError):
            messagebox.showerror("Error", "Could not get employee ID from selection.", parent=self.root)
            return

        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete employee {emp_id} ({emp_name})?\nThis action cannot be undone.", parent=self.root):
            if self.emp_db.delete_employee(emp_id):
                self.refresh_employee_list()
                messagebox.showinfo("Success", f"Employee {emp_id} deleted.", parent=self.root)
            else:
                messagebox.showerror("Error", f"Failed to delete employee {emp_id}.", parent=self.root)

    def toggle_tracking(self):
        """Start or stop the video tracking thread."""
        if self.is_tracking:
            self.is_tracking = False
            self.track_btn.config(text="Start Tracking")
            self.status_var.set("Tracking stopped.")
            logging.info("Tracking stopped by user.")
            # The thread will exit its loop naturally
            # Reset trackers
            self.trackers = {}
            self.next_track_id = 0
            if self.capture_thread and self.capture_thread.is_alive():
                 # Give thread a moment to finish current frame
                 self.capture_thread.join(timeout=0.5) 
        else:
            # Check if camera is accessible before starting thread
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                messagebox.showerror("Camera Error", f"Could not open video source {self.video_source}.")
                logging.error(f"Failed to open video source {self.video_source}.")
                cap.release()
                return
            cap.release() # Release test capture
            
            self.is_tracking = True
            self.track_btn.config(text="Stop Tracking")
            self.status_var.set("Tracking active...")
            logging.info("Tracking started.")
            # Clear previous trackers before starting
            self.trackers = {}
            self.next_track_id = 0
            # Start tracking thread
            self.capture_thread = Thread(target=self.tracking_loop, daemon=True)
            self.capture_thread.start()

    def tracking_loop(self):
        """Main loop for video capture, processing, and tracking."""
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            logging.error(f"Camera {self.video_source} disappeared or failed after check.")
            # Schedule UI update from main thread
            self.root.after(0, lambda: self.status_var.set("Error: Camera disconnected."))
            self.root.after(0, lambda: self.track_btn.config(text="Start Tracking"))
            self.is_tracking = False
            return

        fps_time = time.time()
        frame_count = 0

        while self.is_tracking:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab frame from camera.")
                time.sleep(0.1) # Avoid busy-waiting if camera fails temporarily
                continue

            h, w = frame.shape[:2]
            
            # --- Pose Detection ---
            try:
                landmarks_results = self.pose_detector.process_frame(frame) # Returns a list of pose_landmarks objects
            except Exception as e:
                 logging.error(f"Error during pose detection: {e}")
                 landmarks_results = [] # Continue without detections this frame

            # --- Tracker Update and Association ---
            current_detections = [] # List of (bbox, landmarks_object)
            for landmarks in landmarks_results:
                bbox = get_person_bbox(landmarks.landmark, frame.shape)
                current_detections.append({'bbox': bbox, 'landmarks': landmarks})
                # Draw raw landmarks immediately for visual feedback
                frame = self.pose_detector.draw_landmarks(frame, landmarks)

            # --- Match detections to existing trackers ---
            matched_track_ids = set()
            unmatched_detections = list(range(len(current_detections)))

            # Try matching existing trackers first
            for track_id, tracker in list(self.trackers.items()): # Use list copy for safe deletion
                 # Predict tracker's next position (optional, simple: use last bbox)
                 best_match_idx = -1
                 max_iou = IOU_THRESHOLD # Use configured threshold as minimum

                 # Find best overlapping detection for this tracker
                 for i, det_idx in enumerate(unmatched_detections):
                     detection = current_detections[det_idx]
                     overlaps, iou = check_overlap(tracker.bbox, detection['bbox'])
                     if overlaps and iou > max_iou:
                         max_iou = iou
                         best_match_idx = i # Index within unmatched_detections list

                 if best_match_idx != -1:
                     # Match found: Update tracker
                     matched_det_idx = unmatched_detections.pop(best_match_idx)
                     matched_detection = current_detections[matched_det_idx]
                     
                     tracker.update_position(matched_detection['bbox'])
                     tracker.add_landmarks(matched_detection['landmarks'], frame.shape)
                     matched_track_ids.add(track_id)
                 else:
                      # No good match found for this tracker, check timeout
                      if time.time() - tracker.last_seen > 3.0: # Timeout after 3 seconds
                          logging.info(f"Tracker {track_id} timed out and removed.")
                          del self.trackers[track_id]

            # --- Create new trackers for unmatched detections ---
            for det_idx in unmatched_detections:
                if len(self.trackers) < MAX_PEOPLE_TRACK:
                     detection = current_detections[det_idx]
                     new_tracker = EnhancedPersonTracker(self.next_track_id, detection['bbox'])
                     new_tracker.add_landmarks(detection['landmarks'], frame.shape)
                     self.trackers[self.next_track_id] = new_tracker
                     logging.info(f"New tracker created: ID {self.next_track_id}")
                     self.next_track_id += 1
                else:
                     logging.warning("Maximum number of trackers reached.")
                     break # Stop creating new trackers

            # --- Process Trackers (Registration & Identification) ---
            for track_id, tracker in self.trackers.items():
                x_min, y_min, x_max, y_max, center_x, center_y = tracker.bbox
                box_color = (255, 150, 0) # Default color (Orange)
                label = f"TrackID: {track_id}"
                
                # --- Registration Logic ---
                if self.registration_mode and self.registering_employee is not None:
                    # Check if this tracker is the one being registered (optional, assumes one person for registration)
                    # Check centrality
                    frame_center_x = w / 2
                    is_centered = abs(center_x - frame_center_x) < (w * REGISTRATION_CENTER_THRESHOLD_FACTOR)
                    
                    if is_centered:
                        box_color = (0, 255, 255) # Yellow for potential registration candidate
                        label += " (Centering)"
                        
                        # Check if buffer is full enough and feature is stable
                        identity_feature = tracker.get_identity_feature()
                        if identity_feature is not None:
                            label += " (Ready!)"
                            box_color = (0, 0, 255) # Red for ready to register
                            
                            # --- SAVE FEATURE ---
                            if self.emp_db.update_employee(self.registering_employee, 'enhanced_feature', identity_feature):
                                logging.info(f"Successfully registered features for Employee {self.registering_employee} from Tracker {track_id}.")
                                # Schedule UI updates from the main thread
                                self.root.after(0, lambda: self.status_var.set(f"Features registered for Employee {self.registering_employee}!"))
                                self.root.after(0, self.refresh_employee_list)
                                self.root.after(0, lambda: messagebox.showinfo("Registration Complete", f"Features successfully registered for Employee {self.registering_employee}."))
                            else:
                                 logging.error(f"Failed to save features for Employee {self.registering_employee}.")
                                 self.root.after(0, lambda: self.status_var.set(f"Error saving features for {self.registering_employee}."))

                            # Exit registration mode automatically
                            self.registration_mode = False
                            self.registering_employee = None
                            # No 'break' here if multiple people could be centered

                # --- Identification Logic ---
                elif not self.registration_mode: # Only identify if not registering
                    emp_id, confidence, emp_info = tracker.match_with_database(self.emp_db)
                    
                    if tracker.stable and emp_id is not None and emp_info is not None:
                        box_color = (0, 255, 0) # Green for identified
                        emp_name = emp_info.get('name', 'Unknown')
                        label = f"ID: {emp_id} ({emp_name}) - {confidence:.2f}"
                    elif emp_id is not None: # Match found but not stable yet
                        box_color = (255, 255, 0) # Cyan for tentative match
                        label = f"Matching: {emp_id}? ({confidence:.2f})"
                    # else: label remains TrackID

                # --- Drawing ---
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                # Put label above the box
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_ymin = max(y_min, label_size[1] + 10)
                cv2.rectangle(frame, (x_min, label_ymin - label_size[1] - 10), 
                              (x_min + label_size[0], label_ymin - base_line -10), box_color, cv2.FILLED)
                cv2.putText(frame, label, (x_min, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # --- Display FPS ---
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                frame_count = 0
                fps_time = current_time
            
            # --- Display Registration Mode Banner ---
            if self.registration_mode:
                 cv2.putText(frame, f"REGISTRATION MODE: Employee {self.registering_employee}", 
                             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


            # --- Update Canvas ---
            try:
                 # Resize frame slightly if needed for performance or display fit
                 # frame_resized = cv2.resize(frame, (new_w, new_h)) 
                 cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                 pil_image = Image.fromarray(cv2image)
                 # Resizing for canvas if necessary (can impact quality)
                 # canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
                 # if canvas_w > 10 and canvas_h > 10: # Check if canvas size is valid
                 #    pil_image = pil_image.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)

                 photo = ImageTk.PhotoImage(image=pil_image)
                 # Schedule canvas update in the main thread
                 self.root.after(0, self.update_canvas, photo)
            except Exception as e:
                 logging.error(f"Error converting/displaying frame: {e}")

        # --- Cleanup after loop exit ---
        cap.release()
        self.pose_detector.close() # Release MediaPipe resources
        logging.info("Video capture released and pose detector closed.")
        # Ensure status reflects stopped state
        self.root.after(0, lambda: self.status_var.set("Tracking stopped."))

    def update_canvas(self, photo):
        """Update the Tkinter canvas with the new frame."""
        # Check if canvas exists and is valid
        if self.canvas.winfo_exists():
            # No need to resize canvas here, just update image
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            # Keep a reference to the photo object to prevent garbage collection
            self.canvas.image = photo 
        else:
            logging.warning("Canvas widget no longer exists, cannot update.")

    def visualize_features(self):
        """Callback for the Visualize Features button."""
        if not self.trackers:
            messagebox.showinfo("No Data", "No active trackers to visualize.", parent=self.root)
            return
        
        # Call the internal function to create the plot window
        visualize_features_internal(self.trackers, self.emp_db, self.root)

    def on_close(self):
        """Handle application closing."""
        logging.info("Close request received. Shutting down.")
        # Stop tracking if active
        if self.is_tracking:
            self.is_tracking = False
            if self.capture_thread and self.capture_thread.is_alive():
                 self.capture_thread.join(timeout=1.0) # Wait for thread to finish

        # Save database one last time
        self.emp_db.save()
        logging.info("Employee database saved.")
        
        # Destroy the main window
        self.root.destroy()
        logging.info("Application closed.")


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    # Ensure necessary directories exist (e.g., for logs)
    # if not os.path.exists("logs"): os.makedirs("logs") 

    try:
        root = tk.Tk()
        app = WarehouseTrackingApp(root, video_source=0) # Use 0 for default camera
        root.mainloop()
    except Exception as e:
         logging.critical(f"Unhandled exception in main application: {e}", exc_info=True)
         # Show error to user if possible
         try:
              messagebox.showerror("Critical Error", f"A critical error occurred:\n{e}\n\nCheck the log file '{LOG_FILE}' for details.")
         except: 
              print(f"CRITICAL ERROR: {e}") # Fallback print