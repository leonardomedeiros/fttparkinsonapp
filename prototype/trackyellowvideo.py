import cv2
import numpy as np
cap = cv2.VideoCapture('videotenis.avi')
while(cap.grab()):
    _, frame = cap.retrieve()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([40,255,255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('res',res)

    # Press ESC to close windows
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()