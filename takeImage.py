import numpy as np
import cv2
import time
import pyodbc
import pickle

def take_image():
    global lane_image
    global lane_image_gray
    if cap.isOpened():
        ret, lane_image = cap.read()
        if ret:
            lane_image_gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('C:/BowlingApp/Line2/lane_image.jpg', lane_image)
            print('Save lane')
            cv2.imwrite('C:/BowlingApp/Line2/lane_image_gray.jpg', lane_image_gray)
            print('Save lane gray')

        else:
            print("Error opening video stream or file")
    else:
        print("Not Found")

cap = cv2.VideoCapture('rtsp://192.168.1.33/id=0')#ip_address) 
print(cap)
take_image()
