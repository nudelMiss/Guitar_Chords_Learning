import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera!")
        return
    
    while True: # infinite loop 
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame!")
            break

        cv2.imshow("Camera", frame) # display the frame

        if cv2.waitKey(1) & 0xFF == ord('q'): # exit on 'q' key press
            break

    cap.release()   
    cv2.destroyAllWindows()

if __name__ == "__main__": # run the main function
        main()
    