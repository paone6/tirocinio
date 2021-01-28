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
import argparse
from collections import OrderedDict


# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

#return euclides distance between two points
def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/paone/Desktop/VideoDataset/shape_predictor_68_face_landmarks.dat")

# Opens the Video file
path = "/Users/paone/Desktop/prova_video"
i=0
cosa = None
for videoFile in os.listdir(path):
    cap= cv2.VideoCapture(path + "/" + videoFile)

    while(cap.isOpened()):
        ret, image = cap.read()


        if ret == False:
            break
        # load the input image, resize it, and convert it to grayscale
        
            #image = cv2.imread(frame)
            #image = imutils.resize(image, width=500)
            #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale image
            
        rects = detector(image, 1)
        # loop over the face detections
            

            
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(image, rect)
            shape = shape_to_np(shape)
                


            # and draw them on the image
            xm,ym = FACIAL_LANDMARKS_IDXS["mouth"]
            for (x, y) in shape[xm:ym+1]:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            # show the output image with the face detections + facial landmarks
            cv2.imshow("Output", image)
            #cv2.imshow("Prova", visualize_facial_landmarks(gray,shape))
            cv2.waitKey(0)

        i+=1
    
print("Numero frame ", i) 
cap.release()
cv2.destroyAllWindows()