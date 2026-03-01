import cv2
import numpy as np
import math

class HandDetector:
    def __init__(self):
        self.frets = []
        self.strings = []
        
        # Skin color boundaries in HSV color space
        # These values should be calibrated using the calibrate_color.py script
        self.lower_skin = np.array([0, 58, 31], dtype=np.uint8)
        self.upper_skin = np.array([13, 161, 255], dtype=np.uint8)

    def calibrate_frets(self, frets_x):
        self.frets = sorted(list(frets_x))

    def calibrate_strings(self, strings_y):
        self.strings = sorted(list(strings_y))

    def _get_fret_and_string(self, x, y):
        fret_idx = 1
        for fx in self.frets:
            if x > fx:
                fret_idx += 1
                
        string_idx = 1
        for sy in self.strings:
            if y > sy:
                string_idx += 1
                
        return fret_idx, string_idx

    # Detect fingers and determine their fret and string positions
    # Input: frame (BGR image)
    # Output: (num_fingers, [(x, y, fret, string), ...])
    def detect_fingers_and_frets(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, []
            
        hand_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(hand_contour) < 2000:
            return 0, []
            
        M = cv2.moments(hand_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
            
        hull_indices = cv2.convexHull(hand_contour, returnPoints=False)
        if hull_indices is None or len(hull_indices) < 3:
            return 0, []
            
        candidates = []
        
        # --- 1. CURVATURE ANALYSIS (Smart Peak Detection) ---
        # We measure the angle at every peak to see how "sharp" it is.
        for i in range(len(hull_indices)):
            idx = hull_indices[i][0]
            pt = tuple(hand_contour[idx][0])
            
            dist_center = math.sqrt((pt[0] - cx)**2 + (pt[1] - cy)**2)
            
            # Ignore points too close to the wrist/center
            if dist_center < 40 or pt[1] > cy + 120:
                continue
                
            # Walk 25 points backward and forward along the contour
            step = 25
            idx_prev = (idx - step) % len(hand_contour)
            idx_next = (idx + step) % len(hand_contour)
            
            pt_prev = tuple(hand_contour[idx_prev][0])
            pt_next = tuple(hand_contour[idx_next][0])
            
            # Calculate the angle of the peak using the Law of Cosines
            a = math.sqrt((pt_next[0] - pt_prev[0])**2 + (pt_next[1] - pt_prev[1])**2)
            b = math.sqrt((pt[0] - pt_prev[0])**2 + (pt[1] - pt_prev[1])**2)
            c = math.sqrt((pt_next[0] - pt[0])**2 + (pt_next[1] - pt[1])**2)
            
            if b * c == 0:
                continue
                
            # Math protection for floating point inaccuracies
            val = (b**2 + c**2 - a**2) / (2 * b * c)
            val = max(min(val, 1.0), -1.0) 
            angle = math.acos(val) * 180 / math.pi
            
            # A true finger or knuckle will have an angle less than ~110 degrees
            # The flat edge of a palm will be ~150-180 degrees
            if angle < 110:
                candidates.append((pt, angle))
                
        # --- 2. FILTER DUPLICATES & KEEP THE SHARPEST ---
        filtered_peaks = []
        for pt, angle in candidates:
            is_duplicate = False
            for i, (f_pt, f_angle) in enumerate(filtered_peaks):
                dist = math.sqrt((pt[0] - f_pt[0])**2 + (pt[1] - f_pt[1])**2)
                
                # If points are close, they belong to the same finger
                if dist < 45: 
                    is_duplicate = True
                    # Keep the one that is "sharper" (smaller angle)
                    if angle < f_angle:
                        filtered_peaks[i] = (pt, angle)
                    break
                    
            if not is_duplicate:
                filtered_peaks.append((pt, angle))
                
        # --- 3. TAKE MAX 5 POINTS ---
        # Sort by angle ascending (sharpest peaks first)
        filtered_peaks.sort(key=lambda x: x[1])
        
        # Keep exactly the top 5 sharpest points
        best_5_peaks = filtered_peaks[:5]
        
        # Extract just the (x, y) coordinates (discard the angle data)
        final_points = [x[0] for x in best_5_peaks]
        
        # --- 4. SORT LEFT TO RIGHT FOR LABELING ---
        final_points.sort(key=lambda p: p[0])
        
        points_data = []
        for tip in final_points:
            fret, string = self._get_fret_and_string(tip[0], tip[1])
            # Return raw data: (x, y, fret, string)
            points_data.append((tip[0], tip[1], fret, string))
            
        return len(final_points), points_data