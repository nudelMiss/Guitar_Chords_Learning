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

    def detect_fingers_and_frets(self, frame):
        #fixed the color of the frame and making a mask to detect the hand and fingers
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
            
        # 1. Find the Centroid of the hand
        M = cv2.moments(hand_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
            
        hull_indices = cv2.convexHull(hand_contour, returnPoints=False)
        hull_points = cv2.convexHull(hand_contour, returnPoints=True)
        
        if len(hull_indices) < 3:
            return 0, []
            
        try:
            defects = cv2.convexityDefects(hand_contour, hull_indices)
        except Exception:
            return 0, []
            
        fingertips = []
        
        # 2. Add Topmost point
        topmost = tuple(hand_contour[hand_contour[:, :, 1].argmin()][0])
        fingertips.append(topmost)
        
        # 3. Add peaks from Defects (Valleys)
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
                
                if angle <= 130 and d > 5000:
                    fingertips.append(start)
                    fingertips.append(end)
                    
        # 4. Add peaks from Convex Hull
        for point in hull_points:
            pt = tuple(point[0])
            dist_center = math.sqrt((pt[0] - cx)**2 + (pt[1] - cy)**2)
            
            # INCREASED to cy + 150 because the thumb joint is lower towards the wrist
            if dist_center > 40 and pt[1] < cy + 150:
                fingertips.append(pt)
                    
        # 5. Add Leftmost and Rightmost points
        leftmost = tuple(hand_contour[hand_contour[:, :, 0].argmin()][0])
        rightmost = tuple(hand_contour[hand_contour[:, :, 0].argmax()][0])
        
        dist_left = math.sqrt((leftmost[0] - cx)**2 + (leftmost[1] - cy)**2)
        dist_right = math.sqrt((rightmost[0] - cx)**2 + (rightmost[1] - cy)**2)
        
        # INCREASED to cy + 150 here as well to catch the thumb
        if dist_left > 30 and leftmost[1] < cy + 150:
            fingertips.append(leftmost)
        if dist_right > 30 and rightmost[1] < cy + 150:
            fingertips.append(rightmost)
            
        # 6. Filter duplicates
        filtered_tips = []
        for tip in fingertips:
            is_duplicate = False
            for filtered_tip in filtered_tips:
                dist = math.sqrt((tip[0] - filtered_tip[0])**2 + (tip[1] - filtered_tip[1])**2)
                if dist < 30:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_tips.append(tip)

        # 7. Sort points from left to right (X-axis)
        filtered_tips.sort(key=lambda p: p[0])
        
        points_data = []
        for tip in filtered_tips:
            fret, string = self._get_fret_and_string(tip[0], tip[1])
            # Return raw data: (x, y, fret, string)
            points_data.append((tip[0], tip[1], fret, string))
            
        return len(filtered_tips), points_data