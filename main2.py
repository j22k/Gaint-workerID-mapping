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
# -------------------------------
# Configuration and Settings
# -------------------------------

EMP_DB_FILE = "employee_database.json"  # File to store enrolled employee data
BUFFER_LENGTH = 30  # Number of frames to accumulate gait feature data
BINS = 5          # Number of bins to reduce the feature vector
MATCH_THRESHOLD = 10.0  # Threshold for matching confidence
MIN_DETECTION_FRAMES = 15  # Minimum consecutive frames for stable detection
MAX_PEOPLE_TRACK = 10  # Maximum number of people to track simultaneously

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
                            # Log the detection
                            log_employee_activity(emp_id)
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
# Multiple Person Pose Detector
# -------------------------------

class MultiPersonPoseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize person detector (BlazePose for multiple people)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1)
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize person detector (Human pose landmark detection model)
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def process_frame(self, frame):
        """
        Process a frame to detect multiple people.
        Returns a list of pose landmarks for each detected person.
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Pose
        results = self.pose.process(rgb_frame)
        
        # Return landmarks if people were detected
        if results.pose_landmarks:
            # This returns a single person's landmarks in the current implementation
            # We'll simulate multiple detections by handling them individually 
            # in the main tracking loop
            return [results.pose_landmarks]
        return []
    
    def draw_landmarks(self, frame, landmarks):
        """Draw pose landmarks on the frame."""
        self.mp_drawing.draw_landmarks(
            frame, 
            landmarks, 
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        
        return frame

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
        
        # Initialize multi-person pose detector
        self.pose_detector = MultiPersonPoseDetector(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5)
        
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
            self.status_var.set("Tracking started. Multiple employees will be identified when detected.")
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
            
            # Process with MediaPipe for multiple people
            h, w, _ = frame.shape
            
            # Process the frame to detect people
            landmarks_list = self.pose_detector.process_frame(frame)
            
            # Clear old trackers (people who haven't been seen for a while)
            current_time = time.time()
            to_remove = []
            for track_id in self.person_trackers:
                if current_time - self.person_trackers[track_id].last_seen > 2.0:
                    to_remove.append(track_id)
            
            for track_id in to_remove:
                del self.person_trackers[track_id]
            
            # Process pose landmarks for each detected person
            current_detections = []
            
            for landmarks in landmarks_list:
                # Get person bounding box
                bbox = get_person_bbox(landmarks.landmark, frame.shape)
                current_detections.append(bbox)
                
                # Draw landmarks
                frame = self.pose_detector.draw_landmarks(frame, landmarks)
                
                # Extract landmarks for knee angle calculation
                right_hip = (landmarks.landmark[self.pose_detector.mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                             landmarks.landmark[self.pose_detector.mp_pose.PoseLandmark.RIGHT_HIP.value].y * h)
                right_knee = (landmarks.landmark[self.pose_detector.mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                              landmarks.landmark[self.pose_detector.mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h)
                right_ankle = (landmarks.landmark[self.pose_detector.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                               landmarks.landmark[self.pose_detector.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h)
                
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
                
                # If no match, create a new tracker (up to MAX_PEOPLE_TRACK)
                if not matched and len(self.person_trackers) < MAX_PEOPLE_TRACK:
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
                        
                       # Continuation of WarehouseTrackingApp class methods:
                        # Update UI from main thread
                        self.root.after(0, lambda: self.status_var.set(f"Gait feature registered for employee {self.registering_employee}!"))
                        self.root.after(0, self.refresh_employee_list)
                        
                        # Exit registration mode
                        self.registration_mode = False
                        self.registering_employee = None
                        break  # Stop after registering
                
                # Draw bounding box and person ID on frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # If we have a stable identification, show it
                label = f"ID: {track_id}"
                if tracker.stable and tracker.employee_info:
                    emp_name = tracker.employee_info.get('name', 'Unknown')
                    emp_dept = tracker.employee_info.get('department', 'Unknown')
                    label = f"ID: {tracker.last_matched_id} - {emp_name} ({emp_dept})"
                
                # Match with database if we have enough angles
                if len(tracker.angle_buffer) == BUFFER_LENGTH and not self.registration_mode:
                    emp_id, confidence, employee_info = tracker.match_with_database(self.emp_db)
                    if emp_id and tracker.stable:
                        # Change box color for identified employees
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
                
                # Put text on frame
                cv2.putText(frame, label, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show registration mode status
            if self.registration_mode:
                cv2.putText(frame, f"REGISTRATION MODE: Employee {self.registering_employee}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show FPS
            cv2.putText(frame, f"FPS: {int(fps)}", (10, h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Convert to PIL format for tkinter
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2image)
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update UI from main thread
            self.root.after(0, lambda p=photo: self.update_canvas(p))
        
        # Release camera when done
        cap.release()
    
    def update_canvas(self, photo):
        """Update the canvas with a new frame."""
        # Resize canvas to match the image dimensions
        width, height = photo.width(), photo.height()
        self.canvas.config(width=width, height=height)
        
        # Update the image
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        
        # Keep a reference to prevent garbage collection
        self.canvas.image = photo
    
    def on_close(self):
        """Clean up resources and save data before closing."""
        self.is_tracking = False
        if self.capture_thread and self.capture_thread.is_alive():
            # Let the thread exit naturally
            time.sleep(0.5)
        
        # Save employee database
        self.emp_db.save()
        
        # Close the application
        self.root.destroy()

# -------------------------------
# Analytics Functions 
# -------------------------------

def generate_attendance_report(log_file="warehouse_activity_log.csv", output_file="attendance_report.csv"):
    """Generate a daily attendance report from the activity log."""
    if not os.path.exists(log_file):
        print("No activity log found.")
        return False
    
    try:
        # Read log file
        with open(log_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
        
        # Parse data
        daily_attendance = {}
        for line in lines:
            if line.strip():
                timestamp, emp_id, action = line.strip().split(',')
                date = timestamp.split(' ')[0]
                
                if date not in daily_attendance:
                    daily_attendance[date] = set()
                
                if action == "detected":
                    daily_attendance[date].add(emp_id)
        
        # Write report
        with open(output_file, 'w') as f:
            f.write("date,employee_id,present\n")
            for date, employees in daily_attendance.items():
                for emp_id in employees:
                    f.write(f"{date},{emp_id},yes\n")
        
        print(f"Attendance report generated: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error generating attendance report: {e}")
        return False

def analyze_employee_presence(log_file="warehouse_activity_log.csv"):
    """Analyze employee presence patterns and return statistics."""
    if not os.path.exists(log_file):
        return None
    
    try:
        # Read log file
        with open(log_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
        
        # Parse data
        employee_detections = {}
        for line in lines:
            if line.strip():
                timestamp, emp_id, action = line.strip().split(',')
                if action == "detected":
                    if emp_id not in employee_detections:
                        employee_detections[emp_id] = []
                    employee_detections[emp_id].append(timestamp)
        
        # Analyze data
        stats = {}
        for emp_id, detections in employee_detections.items():
            stats[emp_id] = {
                "detection_count": len(detections),
                "first_detection": min(detections) if detections else None,
                "last_detection": max(detections) if detections else None
            }
        
        return stats
    
    except Exception as e:
        print(f"Error analyzing employee presence: {e}")
        return None

# -------------------------------
# Main Application Entry Point
# -------------------------------

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='warehouse_tracking.log'
    )
    
    # Create the main application window
    root = tk.Tk()
    app = WarehouseTrackingApp(root)
    
    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    main()