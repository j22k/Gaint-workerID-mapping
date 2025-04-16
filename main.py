# import cv2
# import mediapipe as mp
# import numpy as np
# import json
# import os
# from collections import deque
# import time

# # -------------------------------
# # Configuration and Settings
# # -------------------------------

# EMP_DB_FILE = "employee_templates.json"  # File to store enrolled employee templates
# BUFFER_LENGTH = 30  # Number of frames to accumulate gait feature data
# BINS = 5          # Number of bins to reduce the feature vector
# MATCH_THRESHOLD = 10.0  # Threshold for matching confidence
# MIN_DETECTION_FRAMES = 15  # Minimum consecutive frames for stable detection

# # -------------------------------
# # Helper Functions
# # -------------------------------

# def load_employee_db(db_file):
#     """Load employee templates from a JSON file."""
#     if os.path.exists(db_file):
#         with open(db_file, "r") as f:
#             data = json.load(f)
#         # Convert stored lists to numpy arrays
#         for emp in data:
#             data[emp] = np.array(data[emp])
#         return data
#     else:
#         return {}

# def save_employee_db(db_file, db):
#     """Save employee templates to a JSON file."""
#     # Convert numpy arrays to lists for JSON serialization
#     serializable_db = {emp: db[emp].tolist() for emp in db}
#     with open(db_file, "w") as f:
#         json.dump(serializable_db, f, indent=4)

# def compute_angle(a, b, c):
#     """
#     Compute the angle (in degrees) between three points: a, b, c 
#     with b as the vertex. Points are (x, y) tuples.
#     """
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     ba = a - b
#     bc = c - b
#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
#     angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
#     return np.degrees(angle)

# def process_buffer(angle_buffer, bins):
#     """
#     Process the buffer (list of angles) into a feature vector.
#     Here we split the buffer into 'bins' segments and average.
#     """
#     arr = np.array(angle_buffer)
#     # If the buffer length is not exactly divisible by bins, truncate the extra values
#     seg_length = len(arr) // bins
#     feature = []
#     for i in range(bins):
#         seg = arr[i * seg_length: (i+1) * seg_length]
#         feature.append(np.mean(seg))
#     return np.array(feature)

# def match_employee(gait_feature, emp_db, threshold=MATCH_THRESHOLD):
#     """
#     Match the current gait feature with the enrolled employee templates.
#     Returns the best matched employee ID if the distance is below the threshold;
#     otherwise, returns None.
#     """
#     best_match = None
#     min_dist = float('inf')
#     for emp_id, template in emp_db.items():
#         dist = np.linalg.norm(gait_feature - template)
#         if dist < min_dist:
#             min_dist = dist
#             best_match = emp_id
#     # Only return a match if the distance is below a set threshold.
#     if min_dist < threshold:
#         return best_match, min_dist
#     return None, min_dist

# def get_person_bbox(landmarks, frame_shape):
#     """
#     Get the bounding box coordinates for a person based on pose landmarks.
#     Returns (x_min, y_min, x_max, y_max, center_x, center_y)
#     """
#     h, w = frame_shape[:2]
#     x_coords = [lm.x * w for lm in landmarks]
#     y_coords = [lm.y * h for lm in landmarks]
    
#     # Add padding to the bounding box
#     padding = 0.1
#     x_min = max(0, int(min(x_coords) - padding * w))
#     y_min = max(0, int(min(y_coords) - padding * h))
#     x_max = min(w, int(max(x_coords) + padding * w))
#     y_max = min(h, int(max(y_coords) + padding * h))
    
#     center_x = (x_min + x_max) // 2
#     center_y = (y_min + y_max) // 2
    
#     return (x_min, y_min, x_max, y_max, center_x, center_y)

# def check_overlap(bbox1, bbox2, iou_threshold=0.3):
#     """
#     Check if two bounding boxes overlap significantly.
#     Returns True if IoU is above threshold.
#     """
#     # Extract coordinates
#     x1_min, y1_min, x1_max, y1_max = bbox1[:4]
#     x2_min, y2_min, x2_max, y2_max = bbox2[:4]
    
#     # Calculate intersection area
#     x_left = max(x1_min, x2_min)
#     y_top = max(y1_min, y2_min)
#     x_right = min(x1_max, x2_max)
#     y_bottom = min(y1_max, y2_max)
    
#     if x_right < x_left or y_bottom < y_top:
#         return False
    
#     intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
#     # Calculate union area
#     box1_area = (x1_max - x1_min) * (y1_max - y1_min)
#     box2_area = (x2_max - x2_min) * (y2_max - y2_min)
#     union_area = box1_area + box2_area - intersection_area
    
#     # Calculate IoU
#     iou = intersection_area / union_area if union_area > 0 else 0
    
#     return iou > iou_threshold

# # -------------------------------
# # Initialize Modules and Databases
# # -------------------------------

# # Load or create employee database
# employee_db = load_employee_db(EMP_DB_FILE)

# # Initialize Mediapipe pose estimation with higher confidence for multi-person
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.7, 
#                     min_tracking_confidence=0.5,
#                     model_complexity=1)
# mp_drawing = mp.solutions.drawing_utils

# # Class to track each person in the scene
# class PersonTracker:
#     def __init__(self, track_id, bbox):
#         self.track_id = track_id
#         self.bbox = bbox
#         self.angle_buffer = deque(maxlen=BUFFER_LENGTH)
#         self.last_matched_id = None
#         self.match_confidence = 0
#         self.consecutive_matches = 0
#         self.last_seen = time.time()
#         self.stable = False
    
#     def update_position(self, bbox):
#         self.bbox = bbox
#         self.last_seen = time.time()
    
#     def add_angle(self, angle):
#         self.angle_buffer.append(angle)
        
#     def match_with_database(self, emp_db):
#         if len(self.angle_buffer) >= BUFFER_LENGTH:
#             feature = process_buffer(self.angle_buffer, BINS)
#             emp_id, confidence = match_employee(feature, emp_db)
            
#             if emp_id is not None and emp_id == self.last_matched_id:
#                 self.consecutive_matches += 1
#                 if self.consecutive_matches >= MIN_DETECTION_FRAMES:
#                     self.stable = True
#             else:
#                 self.consecutive_matches = 1
#                 self.stable = False
            
#             self.last_matched_id = emp_id
#             self.match_confidence = confidence
            
#             return emp_id, confidence
#         return None, float('inf')

# # Person trackers storage
# person_trackers = {}
# next_track_id = 0

# # Log file for warehouse activity
# log_file = "warehouse_activity_log.csv"
# if not os.path.exists(log_file):
#     with open(log_file, 'w') as f:
#         f.write("timestamp,employee_id,action\n")

# def log_employee_activity(emp_id, action="detected"):
#     """Log employee activity to CSV file"""
#     timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
#     with open(log_file, 'a') as f:
#         f.write(f"{timestamp},{emp_id},{action}\n")

# # -------------------------------
# # Main Video Capture and Processing Loop
# # -------------------------------

# print("Warehouse Employee Tracking System")
# print("==================================")
# print("Press 'e' to enroll a new employee based on current gait feature.")
# print("Press 's' to save the current employee database.")
# print("Press 'l' to list all enrolled employees.")
# print("Press 'q' to quit.")

# # Open video capture - change to appropriate source for warehouse camera
# cap = cv2.VideoCapture(0)

# # For FPS calculation
# prev_frame_time = 0
# new_frame_time = 0

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture video. Exiting...")
#         break
    
#     # Calculate FPS
#     new_frame_time = time.time()
#     fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
#     prev_frame_time = new_frame_time
    
#     # Convert to RGB for MediaPipe
#     h, w, _ = frame.shape
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     # Process pose detection
#     results = pose.process(rgb_frame)
    
#     # Clear old trackers that haven't been seen recently
#     current_time = time.time()
#     to_remove = []
#     for track_id in person_trackers:
#         if current_time - person_trackers[track_id].last_seen > 2.0:  # 2 seconds timeout
#             to_remove.append(track_id)
    
#     for track_id in to_remove:
#         del person_trackers[track_id]
    
#     # List to store current detections
#     current_detections = []
    
#     # Process multiple people if detected
#     if results.pose_landmarks:
#         # For single person pose estimation, convert to list format for consistent processing
#         pose_landmarks_list = [results.pose_landmarks]
        
#         for landmarks in pose_landmarks_list:
#             # Get person bounding box
#             bbox = get_person_bbox(landmarks.landmark, frame.shape)
#             current_detections.append(bbox)
            
#             # Draw landmarks on the frame
#             mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)
            
#             # Extract landmarks for right hip, knee, and ankle
#             right_hip = (landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
#                          landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h)
#             right_knee = (landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
#                           landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h)
#             right_ankle = (landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
#                            landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h)
            
#             # Compute knee angle
#             knee_angle = compute_angle(right_hip, right_knee, right_ankle)
            
#             # Check if this person matches an existing tracker
#             matched = False
#             for track_id, tracker in person_trackers.items():
#                 if check_overlap(bbox, tracker.bbox):
#                     # Update existing tracker
#                     tracker.update_position(bbox)
#                     tracker.add_angle(knee_angle)
#                     matched = True
#                     break
            
#             # If no match, create a new tracker
#             if not matched:
#                 new_tracker = PersonTracker(next_track_id, bbox)
#                 new_tracker.add_angle(knee_angle)
#                 person_trackers[next_track_id] = new_tracker
#                 next_track_id += 1
    
#     # Process all current trackers for identification
#     for track_id, tracker in person_trackers.items():
#         x_min, y_min, x_max, y_max, _, _ = tracker.bbox
        
#         # Run matching algorithm if we have enough frames
#         emp_id, confidence = tracker.match_with_database(employee_db)
        
#         # Determine display information
#         if tracker.stable and tracker.last_matched_id:
#             id_text = f"ID: {tracker.last_matched_id}"
#             color = (0, 255, 0)  # Green for confident match
            
#             # Log stable detections (only log once per detection sequence)
#             if tracker.consecutive_matches == MIN_DETECTION_FRAMES:
#                 log_employee_activity(tracker.last_matched_id)
#         elif tracker.last_matched_id:
#             id_text = f"ID: {tracker.last_matched_id}?"
#             color = (0, 255, 255)  # Yellow for tentative match
#         else:
#             id_text = "Unknown"
#             color = (0, 0, 255)  # Red for unknown
        
#         # Draw bounding box and ID
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
#         cv2.putText(frame, id_text, (x_min, y_min - 10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
#         # Show buffer fill status
#         buffer_text = f"Buffer: {len(tracker.angle_buffer)}/{BUFFER_LENGTH}"
#         cv2.putText(frame, buffer_text, (x_min, y_max + 20), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
#     # Display number of people detected
#     people_count = len(person_trackers)
#     cv2.putText(frame, f"People: {people_count}", (10, 30), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
#     # Display FPS
#     cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 30), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
#     # Show the frame
#     cv2.imshow("Warehouse Employee Tracking", frame)
    
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("e"):
#         # Find the most centered person for enrollment
#         center_person_id = None
#         min_distance = float('inf')
#         frame_center_x, frame_center_y = w//2, h//2
        
#         for track_id, tracker in person_trackers.items():
#             _, _, _, _, center_x, center_y = tracker.bbox
#             distance = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
            
#             if distance < min_distance:
#                 min_distance = distance
#                 center_person_id = track_id
        
#         if center_person_id is not None and len(person_trackers[center_person_id].angle_buffer) == BUFFER_LENGTH:
#             new_feature = process_buffer(person_trackers[center_person_id].angle_buffer, BINS)
#             emp_id = input("Enter new employee ID: ").strip()
#             if emp_id:
#                 employee_db[emp_id] = new_feature
#                 save_employee_db(EMP_DB_FILE, employee_db)
#                 print(f"Enrolled employee {emp_id} with feature: {new_feature}")
#                 log_employee_activity(emp_id, "enrolled")
#         else:
#             print("Not enough data to enroll. Ensure person is centered and walking.")
    
#     elif key == ord("s"):
#         save_employee_db(EMP_DB_FILE, employee_db)
#         print(f"Saved {len(employee_db)} employee templates to {EMP_DB_FILE}")
    
#     elif key == ord("l"):
#         print("\nEnrolled Employees:")
#         for i, emp_id in enumerate(employee_db):
#             print(f"{i+1}. ID: {emp_id}")
#         print()
    
#     elif key == ord("q"):
#         break

# # Clean up
# cap.release()
# cv2.destroyAllWindows()
# print("System shut down.")
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

# -------------------------------
# Configuration and Settings
# -------------------------------

EMP_DB_FILE = "employee_database.json"  # File to store enrolled employee data
BUFFER_LENGTH = 30  # Number of frames to accumulate gait feature data
BINS = 5          # Number of bins to reduce the feature vector
MATCH_THRESHOLD = 10.0  # Threshold for matching confidence
MIN_DETECTION_FRAMES = 15  # Minimum consecutive frames for stable detection

# -------------------------------
# Employee Database Functions
# -------------------------------

class EmployeeDatabase:
    def __init__(self, db_file):
        self.db_file = db_file
        self.employees = self.load()
    
    def load(self):
        """Load employee data from a JSON file."""
        if os.path.exists(self.db_file):
            with open(self.db_file, "r") as f:
                data = json.load(f)
            
            # Convert stored lists to numpy arrays for gait features
            for emp_id in data:
                if 'gait_feature' in data[emp_id]:
                    data[emp_id]['gait_feature'] = np.array(data[emp_id]['gait_feature'])
            return data
        else:
            return {}
    
    def save(self):
        """Save employee data to a JSON file."""
        # Create a copy to avoid modifying the original
        save_data = {}
        for emp_id, emp_data in self.employees.items():
            save_data[emp_id] = emp_data.copy()
            if 'gait_feature' in save_data[emp_id]:
                save_data[emp_id]['gait_feature'] = save_data[emp_id]['gait_feature'].tolist()
        
        with open(self.db_file, "w") as f:
            json.dump(save_data, f, indent=4)
        
        print(f"Employee database saved to {self.db_file}")
    
    def add_employee(self, emp_id, name, department, gait_feature=None):
        """Add a new employee to the database."""
        self.employees[emp_id] = {
            'name': name,
            'department': department,
            'registered_date': time.strftime("%Y-%m-%d"),
            'last_detected': None
        }
        
        if gait_feature is not None:
            self.employees[emp_id]['gait_feature'] = gait_feature
        
        self.save()
        return True
    
    def update_employee(self, emp_id, field, value):
        """Update employee information."""
        if emp_id in self.employees:
            self.employees[emp_id][field] = value
            self.save()
            return True
        return False
    
    def delete_employee(self, emp_id):
        """Remove an employee from the database."""
        if emp_id in self.employees:
            del self.employees[emp_id]
            self.save()
            return True
        return False
    
    def get_employee_info(self, emp_id):
        """Get employee information by ID."""
        return self.employees.get(emp_id, None)
    
    def get_all_employees(self):
        """Get all employees."""
        return self.employees
    
    def has_gait_feature(self, emp_id):
        """Check if employee has a registered gait feature."""
        return emp_id in self.employees and 'gait_feature' in self.employees[emp_id]
    
    def match_employee(self, gait_feature, threshold=MATCH_THRESHOLD):
        """
        Match a gait feature with the employee database.
        Returns matched employee ID and confidence score.
        """
        best_match = None
        min_dist = float('inf')
        
        for emp_id, emp_data in self.employees.items():
            if 'gait_feature' in emp_data:
                template = emp_data['gait_feature']
                dist = np.linalg.norm(gait_feature - template)
                if dist < min_dist:
                    min_dist = dist
                    best_match = emp_id
        
        # Only return a match if the distance is below the threshold
        if min_dist < threshold and best_match is not None:
            return best_match, min_dist
        return None, min_dist

# -------------------------------
# Helper Functions
# -------------------------------

def compute_angle(a, b, c):
    """
    Compute the angle (in degrees) between three points: a, b, c 
    with b as the vertex. Points are (x, y) tuples.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def process_buffer(angle_buffer, bins):
    """
    Process the buffer (list of angles) into a feature vector.
    """
    arr = np.array(angle_buffer)
    seg_length = len(arr) // bins
    feature = []
    for i in range(bins):
        seg = arr[i * seg_length: (i+1) * seg_length]
        feature.append(np.mean(seg))
    return np.array(feature)

def get_person_bbox(landmarks, frame_shape):
    """
    Get the bounding box coordinates for a person based on pose landmarks.
    Returns (x_min, y_min, x_max, y_max, center_x, center_y)
    """
    h, w = frame_shape[:2]
    x_coords = [lm.x * w for lm in landmarks]
    y_coords = [lm.y * h for lm in landmarks]
    
    # Add padding to the bounding box
    padding = 0.1
    x_min = max(0, int(min(x_coords) - padding * w))
    y_min = max(0, int(min(y_coords) - padding * h))
    x_max = min(w, int(max(x_coords) + padding * w))
    y_max = min(h, int(max(y_coords) + padding * h))
    
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    return (x_min, y_min, x_max, y_max, center_x, center_y)

def check_overlap(bbox1, bbox2, iou_threshold=0.3):
    """
    Check if two bounding boxes overlap significantly.
    """
    # Extract coordinates
    x1_min, y1_min, x1_max, y1_max = bbox1[:4]
    x2_min, y2_min, x2_max, y2_max = bbox2[:4]
    
    # Calculate intersection area
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou > iou_threshold

def log_employee_activity(emp_id, action="detected", log_file="warehouse_activity_log.csv"):
    """Log employee activity to CSV file"""
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("timestamp,employee_id,action\n")
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"{timestamp},{emp_id},{action}\n")

# -------------------------------
# Person Tracker Class
# -------------------------------

class PersonTracker:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox
        self.angle_buffer = deque(maxlen=BUFFER_LENGTH)
        self.last_matched_id = None
        self.match_confidence = 0
        self.consecutive_matches = 0
        self.last_seen = time.time()
        self.stable = False
        self.employee_info = None
    
    def update_position(self, bbox):
        self.bbox = bbox
        self.last_seen = time.time()
    
    def add_angle(self, angle):
        self.angle_buffer.append(angle)
        
    def match_with_database(self, emp_db):
        if len(self.angle_buffer) >= BUFFER_LENGTH:
            feature = process_buffer(self.angle_buffer, BINS)
            emp_id, confidence = emp_db.match_employee(feature)
            
            if emp_id is not None:
                if emp_id == self.last_matched_id:
                    self.consecutive_matches += 1
                    if self.consecutive_matches >= MIN_DETECTION_FRAMES:
                        self.stable = True
                        if self.employee_info is None:
                            self.employee_info = emp_db.get_employee_info(emp_id)
                            # Update last detected time
                            emp_db.update_employee(emp_id, 'last_detected', time.strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    self.consecutive_matches = 1
                    self.stable = False
                
                self.last_matched_id = emp_id
                self.match_confidence = confidence
                
                return emp_id, confidence, self.employee_info
            else:
                self.consecutive_matches = 0
                self.stable = False
                self.last_matched_id = None
                
            return None, confidence, None
        return None, float('inf'), None

# -------------------------------
# GUI Application
# -------------------------------

class WarehouseTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Warehouse Employee Tracking System")
        self.root.geometry("1200x700")
        
        # Initialize employee database
        self.emp_db = EmployeeDatabase(EMP_DB_FILE)
        
        # Initialize Mediapipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, 
                                      min_tracking_confidence=0.5,
                                      model_complexity=1)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Tracking variables
        self.person_trackers = {}
        self.next_track_id = 0
        self.is_tracking = False
        self.capture_thread = None
        self.video_source = 0  # Default camera
        
        # Registration mode variables
        self.registration_mode = False
        self.registering_employee = None
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left frame for video feed
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create right frame for controls and employee list
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)
        
        # Create video canvas
        self.canvas = tk.Canvas(self.left_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create control panel
        self.control_frame = ttk.LabelFrame(self.right_frame, text="Controls")
        self.control_frame.pack(fill=tk.X, pady=5)
        
        # Start/Stop tracking button
        self.track_btn = ttk.Button(self.control_frame, text="Start Tracking", command=self.toggle_tracking)
        self.track_btn.pack(fill=tk.X, pady=5, padx=5)
        
        # Registration button
        self.register_btn = ttk.Button(self.control_frame, text="Register New Employee", command=self.show_registration_dialog)
        self.register_btn.pack(fill=tk.X, pady=5, padx=5)
        
        # Employee management section
        self.emp_frame = ttk.LabelFrame(self.right_frame, text="Employee Management")
        self.emp_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create treeview for employee list
        self.tree_columns = ("ID", "Name", "Department", "Registered", "Last Detected")
        self.tree = ttk.Treeview(self.emp_frame, columns=self.tree_columns, show="headings")
        
        # Configure column headings
        for col in self.tree_columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80)
        
        # Add scrollbars
        vsb = ttk.Scrollbar(self.emp_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.emp_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Pack tree and scrollbars
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Add context menu for employees
        self.tree.bind("<Button-3>", self.show_employee_context_menu)
        
        # Load employee data into treeview
        self.refresh_employee_list()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("System ready. Click 'Start Tracking' to begin.")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Set up cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def refresh_employee_list(self):
        """Refresh the employee treeview with current data."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add employees to treeview
        for emp_id, emp_data in self.emp_db.get_all_employees().items():
            values = (
                emp_id,
                emp_data.get('name', ''),
                emp_data.get('department', ''),
                emp_data.get('registered_date', ''),
                emp_data.get('last_detected', 'Never')
            )
            tag = 'registered' if self.emp_db.has_gait_feature(emp_id) else 'unregistered'
            self.tree.insert('', 'end', values=values, tags=(tag,))
        
        # Configure tag appearances
        self.tree.tag_configure('registered', background='#e6ffe6')
        self.tree.tag_configure('unregistered', background='#ffe6e6')
    
    def show_employee_context_menu(self, event):
        """Show context menu for employee management."""
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            menu = tk.Menu(self.root, tearoff=0)
            menu.add_command(label="Edit Employee", command=self.edit_selected_employee)
            menu.add_command(label="Register Gait Feature", command=self.register_gait_for_selected)
            menu.add_command(label="Delete Employee", command=self.delete_selected_employee)
            menu.post(event.x_root, event.y_root)
    
    def edit_selected_employee(self):
        """Edit the selected employee."""
        selected = self.tree.selection()
        if not selected:
            return
        
        item = selected[0]
        emp_id = self.tree.item(item, 'values')[0]
        emp_data = self.emp_db.get_employee_info(emp_id)
        
        if not emp_data:
            return
        
        # Create edit dialog
        edit_dialog = tk.Toplevel(self.root)
        edit_dialog.title(f"Edit Employee: {emp_id}")
        edit_dialog.geometry("300x200")
        edit_dialog.resizable(False, False)
        edit_dialog.grab_set()
        
        # Name field
        ttk.Label(edit_dialog, text="Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        name_var = tk.StringVar(value=emp_data.get('name', ''))
        ttk.Entry(edit_dialog, textvariable=name_var).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Department field
        ttk.Label(edit_dialog, text="Department:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        dept_var = tk.StringVar(value=emp_data.get('department', ''))
        ttk.Entry(edit_dialog, textvariable=dept_var).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Save button
        def save_changes():
            self.emp_db.update_employee(emp_id, 'name', name_var.get())
            self.emp_db.update_employee(emp_id, 'department', dept_var.get())
            self.refresh_employee_list()
            edit_dialog.destroy()
        
        ttk.Button(edit_dialog, text="Save", command=save_changes).grid(row=2, column=0, columnspan=2, pady=10)
    
    def register_gait_for_selected(self):
        """Register gait feature for the selected employee."""
        selected = self.tree.selection()
        if not selected:
            return
        
        item = selected[0]
        emp_id = self.tree.item(item, 'values')[0]
        
        # Check if we're already tracking
        if not self.is_tracking:
            messagebox.showinfo("Start Tracking", "Please start tracking first to capture gait features.")
            return
        
        # Enable registration mode
        self.registration_mode = True
        self.registering_employee = emp_id
        self.status_var.set(f"Registration mode active for employee {emp_id}. Please walk naturally in frame.")
    
    def delete_selected_employee(self):
        """Delete the selected employee."""
        selected = self.tree.selection()
        if not selected:
            return
        
        item = selected[0]
        emp_id = self.tree.item(item, 'values')[0]
        
        # Confirm deletion
        if messagebox.askyesno("Delete Employee", f"Are you sure you want to delete employee {emp_id}?"):
            self.emp_db.delete_employee(emp_id)
            self.refresh_employee_list()
    
    def show_registration_dialog(self):
        """Show dialog to register a new employee."""
        register_dialog = tk.Toplevel(self.root)
        register_dialog.title("Register New Employee")
        register_dialog.geometry("300x200")
        register_dialog.resizable(False, False)
        register_dialog.grab_set()
        
        # Employee ID field
        ttk.Label(register_dialog, text="Employee ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        id_var = tk.StringVar()
        ttk.Entry(register_dialog, textvariable=id_var).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Name field
        ttk.Label(register_dialog, text="Name:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        name_var = tk.StringVar()
        ttk.Entry(register_dialog, textvariable=name_var).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Department field
        ttk.Label(register_dialog, text="Department:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        dept_var = tk.StringVar()
        ttk.Entry(register_dialog, textvariable=dept_var).grid(row=2, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Register button
        def register():
            emp_id = id_var.get().strip()
            name = name_var.get().strip()
            dept = dept_var.get().strip()
            
            if not emp_id or not name:
                messagebox.showerror("Error", "Employee ID and Name are required.")
                return
            
            if emp_id in self.emp_db.get_all_employees():
                messagebox.showerror("Error", f"Employee ID {emp_id} already exists.")
                return
            
            self.emp_db.add_employee(emp_id, name, dept)
            self.refresh_employee_list()
            register_dialog.destroy()
            
            # Ask if they want to register gait feature now
            if messagebox.askyesno("Register Gait Feature", 
                                   f"Employee {emp_id} added. Do you want to register their gait feature now?"):
                if not self.is_tracking:
                    self.toggle_tracking()
                self.registration_mode = True
                self.registering_employee = emp_id
                self.status_var.set(f"Registration mode active for employee {emp_id}. Please walk naturally in frame.")
        
        ttk.Button(register_dialog, text="Register", command=register).grid(row=3, column=0, columnspan=2, pady=10)
    
    def toggle_tracking(self):
        """Toggle the tracking state."""
        if self.is_tracking:
            self.is_tracking = False
            self.track_btn.config(text="Start Tracking")
            self.status_var.set("Tracking stopped.")
            if self.capture_thread and self.capture_thread.is_alive():
                # We don't actually kill the thread, just let the loop exit
                pass
        else:
            self.is_tracking = True
            self.track_btn.config(text="Stop Tracking")
            self.status_var.set("Tracking started. Employees will be identified when detected.")
            self.capture_thread = Thread(target=self.tracking_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
    
    def tracking_loop(self):
        """Main tracking loop that runs in a separate thread."""
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            self.status_var.set("Error: Could not open camera.")
            self.is_tracking = False
            self.track_btn.config(text="Start Tracking")
            return
        
        # For FPS calculation
        prev_frame_time = 0
        
        while self.is_tracking:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time
            
            # Process with MediaPipe
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            # Clear old trackers
            current_time = time.time()
            to_remove = []
            for track_id in self.person_trackers:
                if current_time - self.person_trackers[track_id].last_seen > 2.0:
                    to_remove.append(track_id)
            
            for track_id in to_remove:
                del self.person_trackers[track_id]
            
            # Process pose landmarks if detected
            current_detections = []
            
            if results.pose_landmarks:
                # For single person pose estimation, convert to list format
                pose_landmarks_list = [results.pose_landmarks]
                
                for landmarks in pose_landmarks_list:
                    # Get person bounding box
                    bbox = get_person_bbox(landmarks.landmark, frame.shape)
                    current_detections.append(bbox)
                    
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_pose.POSE_CONNECTIONS)
                    
                    # Extract landmarks for knee angle calculation
                    right_hip = (landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                                 landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * h)
                    right_knee = (landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                                  landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h)
                    right_ankle = (landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                                   landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h)
                    
                    # Compute knee angle
                    knee_angle = compute_angle(right_hip, right_knee, right_ankle)
                    
                    # Update or create tracker
                    matched = False
                    for track_id, tracker in self.person_trackers.items():
                        if check_overlap(bbox, tracker.bbox):
                            # Update existing tracker
                            tracker.update_position(bbox)
                            tracker.add_angle(knee_angle)
                            matched = True
                            break
                    
                    # If no match, create a new tracker
                    if not matched:
                        new_tracker = PersonTracker(self.next_track_id, bbox)
                        new_tracker.add_angle(knee_angle)
                        self.person_trackers[self.next_track_id] = new_tracker
                        self.next_track_id += 1
            
            # Process all current trackers
            for track_id, tracker in self.person_trackers.items():
                x_min, y_min, x_max, y_max, center_x, center_y = tracker.bbox
                
                # Check if we're in registration mode and this is the most centered person
                if self.registration_mode and len(tracker.angle_buffer) == BUFFER_LENGTH:
                    frame_center_x, frame_center_y = w//2, h//2
                    distance = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
                    
                    # If person is centered, use for registration
                    if distance < w//4:  # Use 1/4 of width as threshold
                        feature = process_buffer(tracker.angle_buffer, BINS)
                        self.emp_db.update_employee(self.registering_employee, 'gait_feature', feature)
                        
                        # Log the registration
                        log_employee_activity(self.registering_employee, "gait_registered")
                        
                        # Update UI from main thread
                        self.root.after(0, lambda: self.status_var.set(f"Gait feature registered for employee {self.registering_employee}"))
                        self.root.after(0, self.refresh_employee_list)
                        
                        # Exit registration mode
                        self.registration_mode = False
                        self.registering_employee = None
                        
                        # Add a visual indicator for feedback
                        cv2.putText(frame, "REGISTRATION COMPLETE!", (center_x - 100, center_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Normal identification mode
                else:
                    # Run matching algorithm if we have enough frames
                    emp_id, confidence, emp_info = tracker.match_with_database(self.emp_db)
                    
                    # Determine display information
                    if tracker.stable and tracker.last_matched_id:
                        name = emp_info.get('name', '') if emp_info else ''
                        dept = emp_info.get('department', '') if emp_info else ''
                        id_text = f"ID: {tracker.last_matched_id}"
                        info_text = f"{name}, {dept}"
                        color = (0, 255, 0)  # Green for confident match
                        
                        # Log stable detections (only log once per detection sequence)
                        if tracker.consecutive_matches == MIN_DETECTION_FRAMES:
                            log_employee_activity(tracker.last_matched_id)
                    elif tracker.last_matched_id:
                        id_text = f"ID: {tracker.last_matched_id}?"
                        info_text = "Confirming..."
                        color = (0, 255, 255)  # Yellow for tentative match
                    else:
                        id_text = "Unknown"
                        info_text = ""
                        color = (0, 0, 255)  # Red for unknown
                    
                    # Draw bounding box and ID
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(frame, id_text, (x_min, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Show employee info if available
                    if info_text:
                        cv2.putText(frame, info_text, (x_min, y_max + 20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add FPS and mode info to frame
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if self.registration_mode:
                mode_text = f"REGISTRATION MODE - Employee: {self.registering_employee}"
                cv2.putText(frame, mode_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Convert BGR to RGB for display in tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            
            # Resize to fit canvas if needed
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:  # Ensure canvas has been drawn
                img = img.resize((canvas_width, canvas_height))
            
            # Update canvas with new image
            self.photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
            # Update UI elements in main thread
            current_status = self.status_var.get()
            if "Error" not in current_status and not self.registration_mode:
                active_count = sum(1 for t in self.person_trackers.values() if t.stable)
                self.root.after(0, lambda: self.status_var.set(
                    f"Tracking active. {len(self.person_trackers)} people detected, {active_count} identified."))
        
        # Release resources
        cap.release()
        
        # Update UI in main thread when done
        self.root.after(0, lambda: self.track_btn.config(text="Start Tracking"))
        self.root.after(0, lambda: self.status_var.set("Tracking stopped."))
    
    def generate_report(self):
        """Generate a report of employee activity."""
        report_window = tk.Toplevel(self.root)
        report_window.title("Employee Activity Report")
        report_window.geometry("600x400")
        
        # Create text widget
        report_text = tk.Text(report_window)
        report_text.pack(fill=tk.BOTH, expand=True)
        
        # Try to read log file
        log_file = "warehouse_activity_log.csv"
        if not os.path.exists(log_file):
            report_text.insert(tk.END, "No activity logs found.")
            return
        
        # Parse log file
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Skip header
        if len(lines) > 1:
            lines = lines[1:]
        
        # Process log entries
        entries_by_date = {}
        employees = self.emp_db.get_all_employees()
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                timestamp, emp_id, action = parts[0], parts[1], parts[2]
                date = timestamp.split()[0]
                
                if date not in entries_by_date:
                    entries_by_date[date] = []
                
                emp_name = employees.get(emp_id, {}).get('name', 'Unknown')
                entries_by_date[date].append((timestamp, emp_id, emp_name, action))
        
        # Generate report
        report_text.insert(tk.END, "WAREHOUSE EMPLOYEE ACTIVITY REPORT\n")
        report_text.insert(tk.END, "=" * 50 + "\n\n")
        
        for date in sorted(entries_by_date.keys(), reverse=True):
            report_text.insert(tk.END, f"Date: {date}\n")
            report_text.insert(tk.END, "-" * 50 + "\n")
            
            for entry in sorted(entries_by_date[date]):
                timestamp, emp_id, emp_name, action = entry
                time_only = timestamp.split()[1]
                report_text.insert(tk.END, f"{time_only} - {emp_id} ({emp_name}): {action}\n")
            
            report_text.insert(tk.END, "\n")
        
        # Make text read-only
        report_text.config(state=tk.DISABLED)
    
    def on_close(self):
        """Handle window close event."""
        if self.is_tracking:
            self.is_tracking = False
            
            # Wait for thread to finish
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=1.0)
        
        self.root.destroy()

# -------------------------------
# Main Application Entry Point
# -------------------------------

def main():
    root = tk.Tk()
    app = WarehouseTrackingApp(root)
    
    # Add menu bar
    menubar = tk.Menu(root)
    
    # File menu
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Start/Stop Tracking", command=app.toggle_tracking)
    file_menu.add_command(label="Generate Report", command=app.generate_report)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=app.on_close)
    menubar.add_cascade(label="File", menu=file_menu)
    
    # Employee menu
    emp_menu = tk.Menu(menubar, tearoff=0)
    emp_menu.add_command(label="Register New Employee", command=app.show_registration_dialog)
    emp_menu.add_command(label="Refresh Employee List", command=app.refresh_employee_list)
    menubar.add_cascade(label="Employees", menu=emp_menu)
    
    # Help menu
    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label="About", command=lambda: messagebox.showinfo(
        "About", "Warehouse Employee Tracking System\nVersion 1.0\n\nUses gait recognition to identify employees."))
    menubar.add_cascade(label="Help", menu=help_menu)
    
    root.config(menu=menubar)
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()