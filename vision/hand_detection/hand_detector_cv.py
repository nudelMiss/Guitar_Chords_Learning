import cv2
import numpy as np
import math

class HandDetector:
    def __init__(self):
        self.frets = []
        self.strings = []
        
        # Skin color boundaries in HSV color space
        # These might need tweaking depending on room lighting
        self.lower_skin = np.array([0, 58, 31], dtype=np.uint8)
        self.upper_skin = np.array([13, 161, 255], dtype=np.uint8)

    def calibrate_frets(self, frets_x):
        # Store frets sorted from left to right
        self.frets = sorted(list(frets_x))

    def calibrate_strings(self, strings_y):
        # Store strings sorted from top to bottom
        self.strings = sorted(list(strings_y))

    def _get_fret_and_string(self, x, y):
        # Find which fret the x coordinate belongs to
        fret_idx = 1
        for fx in self.frets:
            if x > fx:
                fret_idx += 1
                
        # Find which string the y coordinate belongs to
        string_idx = 1
        for sy in self.strings:
            if y > sy:
                string_idx += 1
                
        return fret_idx, string_idx

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
            
        # --- 1. מציאת מרכז הכובד של היד (Centroid) ---
        # זה עוזר לנו לדעת איפה האמצע של כף היד
        M = cv2.moments(hand_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
            
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        if len(hull) < 3:
            return 0, []
            
        try:
            defects = cv2.convexityDefects(hand_contour, hull)
        except Exception:
            return 0, []
            
        fingertips = []
        
        # --- 2. הנקודה הכי גבוהה (תמיד תהיה אצבע) ---
        topmost = tuple(hand_contour[hand_contour[:, :, 1].argmin()][0])
        fingertips.append(topmost)
        
        # --- 3. מציאת 4 האצבעות בעזרת "עמקים" (Defects) ---
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(hand_contour[s][0])
                end = tuple(hand_contour[e][0])
                far = tuple(hand_contour[f][0])
                
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                
                angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 180 / math.pi
                
                if angle <= 120 and d > 8000:
                    fingertips.append(start)
                    
        # --- 4. הטריק המתמטי לאגודל: חיפוש בקצוות (Leftmost/Rightmost) ---
        leftmost = tuple(hand_contour[hand_contour[:, :, 0].argmin()][0])
        rightmost = tuple(hand_contour[hand_contour[:, :, 0].argmax()][0])
        
        # נבדוק כמה הקצוות האלו רחוקים ממרכז היד
        dist_left = math.sqrt((leftmost[0] - cx)**2 + (leftmost[1] - cy)**2)
        dist_right = math.sqrt((rightmost[0] - cx)**2 + (rightmost[1] - cy)**2)
        
        # נוסיף את הקצה רק אם הוא רחוק מספיק (מעל 50 פיקסלים מהמרכז) 
        # וגם נמצא מעל שורש כף היד האמיתי (cy + 80)
        if dist_left > 50 and leftmost[1] < cy + 80:
            fingertips.append(leftmost)
            
        if dist_right > 50 and rightmost[1] < cy + 80:
            fingertips.append(rightmost)
            
        # --- 5. סינון כפילויות קרובות ---
        filtered_tips = []
        for tip in fingertips:
            is_duplicate = False
            for filtered_tip in filtered_tips:
                dist = math.sqrt((tip[0] - filtered_tip[0])**2 + (tip[1] - filtered_tip[1])**2)
                if dist < 40:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_tips.append(tip)

        # --- 6. מיון האצבעות ומתן שמות ---
        # ממיינים את הנקודות לפי ציר X (משמאל לימין)
        filtered_tips.sort(key=lambda p: p[0])
        
        # Left hand 
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        
        points_with_names = []
        for idx, tip in enumerate(filtered_tips):
            # 
            name = finger_names[idx] if idx < len(finger_names) else f"Extra_{idx}"
            points_with_names.append((name, tip[0], tip[1]))
            
            # finger name and circle for visualization
            cv2.putText(frame, name, (tip[0] - 20, tip[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.circle(frame, tip, 8, (0, 255, 0), -1)
            
        return len(filtered_tips), points_with_names