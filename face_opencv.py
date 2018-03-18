# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 10:01:45 2018

@author: Mayank
"""

#face detection

import cv2

cascade_face= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cascade_eye= cv2.CascadeClassifier('haarcascade_eye.xml')
#function
def detect(gray,frame):
    faces = cascade_face.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[x:x+w,y:y+h]
        roi_frame =frame[x:x+w,y:y+h]
        eyes = cascade_eye.detectMultiScale(roi_gray,1.1,3)
        for(i,j,ew,eh) in eyes:
            cv2.rectangle(roi_frame,(i,j),(i+ew,j+eh),(0,255,0),2)
    return faces
#video
video_capture = cv2.VideoCapture(0)
while True:
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video',canvas)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
video_capture.release()
cv2.destroyAllWindows()