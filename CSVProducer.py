import cv2
import os
import numpy as np
import glob
import dlib
import csv
import math
import time
import shutil
import psutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class CSVProducer:
    def nothing(self):
        pass
 
# Opens the Video file
cap= cv2.VideoCapture("/Users/paone/Desktop/prova_video/prova.avi")
i=0
cosa = None
while(cap.isOpened()):
    ret, frame = cap.read()


    if ret == False:
        break
    if(i==1):
        cosa = frame
    i+=1
 
print("Numero frame ", i) 
cap.release()
cv2.destroyAllWindows()