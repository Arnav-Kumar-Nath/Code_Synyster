# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 21:52:19 2019

@author: Arnav Kumar Nath
"""
#Importing the library :
import cv2


#Importing the cascades :

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier('haarcascade_eye.xml')



#Defining a function that will do the detection
#Recognition works on gray image ofthe frame
#height=h, with=w
#detectMultiScale scales the height and width
#image will be reduced by 1.3
#We need to have a certain neighbour zone to accept the pixel
#Faces are tuples(it is made up of) of 4 elements(x,y,h,w) to detect the face
def detect(gray,frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)         #5 is the no of neighbour zones
    for (x,y,w,h) in faces:                                     #Looping over the coordinate of the face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)     #(2 is the thickness of the rectangle      
                                                                #x+w and y+h gives the lower right corner coordinate        
                                                                #Rectangle on the face on the desired frame
                                                                #RGB code of colors)))
        roi_gray  = gray[y:y+h,x:x+w]        #Zone of interest of gray image inside the face
        roi_color = frame[y:y+h,x:x+w]         #Region of interest for the original image                                
        eyes      = eye_cascade.detectMultiScale(gray, 1.1, 3)
        for (ex,ey,ew,eh) in eyes:                #Looping over the eye        #eh,ex,ey,ex are the coordinates of the eye rectangle             
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)     #Here frame is the ROI as eyes is spread over the ROI_color frame
    return frame


#Doing some recognition with the webcam
#We need the last frame coming from the webcam
#VideoCapture is a class of cv2 which takes 0 if webcam of laptop,com ;1 for external
video_capture = cv2.VideoCapture(0)
while True:                     #Repeat infinitely till user stops 
    _,frame = video_capture.read()      #'_' will not return anything as read takes two arguments but we want only frame 
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          # this will convert the image to gray (blue, green, red to gray)
   #Canvas is the result of the detected frame
    canvas  = detect(gray, frame)
    cv2.imshow('Video', canvas)         #Display all the frame ina animated way
    if cv2.waitKey(1) & 0xFF == ord('q'):        #ord means order   #To stop the video by pressing q
        break
video_capture.release()             #It releases the webcam
cv2.destroyAllWindow()              #It close all the  window       