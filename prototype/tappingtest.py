# -*- coding: utf-8 -*-
"""
Created on Tue Dec 05 20:56:03 2017

@author: Nobody
"""


#Object Tracking
import cv2
import numpy as np
import math
from datetime import datetime
import matplotlib.pyplot as plt

#-------- functions



def get_angle(points):
    center, p1_abs,p2_abs = points
    
    p1 = []
    p2 = []
    p1.append(p1_abs[0])
    p1.append(p1_abs[1])
    p2.append(p2_abs[0])
    p2.append(p2_abs[1])
    
    p1[0] = p1_abs[0] - ref[0]
    p1[1] = p1_abs[1] - ref[1]
    
    p2[0] = p2_abs[0] - ref[0]
    p2[1] = p2_abs[1] - ref[1]
    
    p1_produto_escalar_p1 =(p1[0]*p2[0]) + (p1[1]*p2[1])
    modulo_p1 = math.sqrt((p1[0]*p1[0]) + (p1[1]*p1[1]))
    modulo_p2 = math.sqrt((p2[0]*p2[0]) + (p2[1]*p2[1]))
    produto = modulo_p1*modulo_p2
    angle = math.acos(p1_produto_escalar_p1/produto)
    #angle = math.atan(p2[1]-p1[1]/p2[0]-p1[0])*(180/3.14)
    return math.degrees(angle)

def get_angle2(points):
    p,p1,p2 = points
    
    p_invertido = p[1]
    p1_invertido = p1[1]
    p2_invertido = p2[1]
    
    m1 = (p_invertido-p1_invertido)/(p1[0]-p[0])
    m2 = (p_invertido-p2_invertido)/(p2[0]-p[0])
    angle2 = abs( (m2 - m1)/(1+(m2*m1)) )
    angle2 = math.atan( angle2 )
    return math.degrees(angle2)     

def draw_lines(points):
    centerPoint = points[0]; 
    for i in range(1, len(points)):
        point = points[i]
        cv2.line(frame, (centerPoint[0], centerPoint[1]), (point[0], point[1]), (255,127,0), 5)                 
    return;

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

print("Finger Tapping Test")
#------------------
# Initalize camera
try:
    #cap = cv2.VideoCapture("finger_tapping1_14_12_Blue.mp4")
    cap = cv2.VideoCapture('E:\\GitHub\\fttparkinsonapp\prototype\\finger_tapping3_15_12_17_Blue.mp4')
    #cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("cars.avi")
except:
    print("Não conseguiu abrir o vídeo")
    cap.release()
    raise
#cap = cv2.VideoCapture(0)
# define range of purple color in HSV
lower_purple = np.array([101,60,0])
upper_purple = np.array([160,255,165])
# globals variables
text = "Posicione sua mao dentro do quadrado - FINGER TAPPING - IFAL 2017"
# Create empty points array
points = []
y_axes  = []
x_axes = []
now = 0
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

# Get default camera window size

ret, frame = cap.read()
Height, Width = frame.shape[:2]
h_center = int(Height/2)
w_center= int(Width/2)
point_left_top_rec = (w_center-150, h_center-150)
point_bottom_right_rec = (w_center+150, h_center+150)
        
frame_count = 0
now = datetime.now()
while (cap.isOpened()):
    
    # Capture webcame frame
    ret, frame = cap.read()
    #transpor (matrix transposta i,j = j,i) imagem e rotaciona-la 90 graus
    frame = cv2.transpose(frame)
    frame = cv2.flip(frame, +1)   
    #equalization
    #img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    #img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    #img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    #
    if ret == False: # se o video tiver no fim saia
        break
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   
    # Threshold the HSV image to get only pink colors
    mask = cv2.inRange(hsv_img, lower_purple, upper_purple)
   
    erode = cv2.erode(mask, None, iterations=2)
    dilation = cv2.dilate(erode, None, iterations=2)
    Gaussian = cv2.GaussianBlur(dilation, (7,7), 0)
    
    maskFinal=Gaussian
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #Gaussian = cv2.GaussianBlur(mask, (3,3), 0)
    # Find contours, OpenCV 3.X users use this line instead
    #  _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _,contours, _ = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #organizando os contornos de maior area para menor
    #sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sorted_contours, boundBoxes = sort_contours(contours)
    # Create empty centre array to store centroid center of mass
    center =   int(Height/2), int(Width/2)

    if len(sorted_contours) > 0:
      
        # Get the largest contour and its center 
        #c = max(contours, key=cv2.contourArea)
        #iterando sobre cada cotorno
        for c in sorted_contours:  
            (x, y), radius = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            try:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            except:
                center =   int(Height/2), int(Width/2)
            # Allow only countors that have a larger than 15 pixel radius
            if radius > 10:
                # Draw cirlce and leave the last center creating a trail
                cv2.circle(frame, (int(x), int(y)), int(radius),(0, 0, 255), 2)
                cv2.circle(frame, center, 2, (0, 255, 0), -1)               
                #cv2.line(frame, (0, 0), center, (255,127,0), 5)
            # Log center points 
            points.append(center)

        draw_lines(points)
            
   
            
    # Display our object tracker
   
    #cv2.rectangle(frame, point_left_top_rec, point_bottom_right_rec, (127,50,127), 2)
    #cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    if len(points) == 3:    
        angle = get_angle2(points)
        cv2.putText(frame, "Angulo:"+str(angle), (70,80), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)   
        y_axes.append(int(angle))
       
        
        
   
    cv2.imshow("Object Tracker", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("closing image", maskFinal)  
    points = []
   
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

#-------plot graphs
    
fig, ax = plt.subplots()
ax.plot(y_axes)
plt.show()
        
#-----------------
        
# Release camera and close any open windows
cap.release()
cv2.destroyAllWindows()