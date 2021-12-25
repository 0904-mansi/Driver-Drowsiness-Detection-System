import numpy as np
import dlib
import cv2

from math import hypot
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def mid(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
count = 0

font = cv2.FONT_HERSHEY_TRIPLEX
while True:
    _, img = cap.read()
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    
if mouth_open_ratio > 0.380 and eye_open_ratio > 4.0 or eye_open_ratio > 4.30:
    count +=1
else:
    count = 0
x,y = face_roi.left(), face_roi.top()
x1,y1 = face_roi.right(), face_roi.bottom()
if count>10:
    cv2.rectangle(img, (x,y), (x1,y1), (0, 0, 255), 2)
    cv2.putText(img, "Sleepy", (x, y-5), font, 0.5, (0, 0, 255))
    
else:
    cv2.rectangle(img, (x,y), (x1,y1), (0, 255, 0), 2)
cap.release()

cv2.destroyAllWindows()
