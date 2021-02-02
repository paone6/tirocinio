# -----------------------------------------------------------------------------
# Questo script, dato un video in input, si occuppa di estrarre il maggior 
# numero possibile di sottovideo di 15 secondi in cui il volto del soggetto
# sia ben visibile e in cui quest'ultimo parli
#
# Mario Paone
# ------------------------------------------------------------------------------

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

detector = dlib.get_frontal_face_detector()  #inizializza il face detector(HOG-based) della libreria dlib
predictor = dlib.shape_predictor("/Users/paone/Desktop/VideoDataset/shape_predictor_68_face_landmarks.dat")  #crea il predictor per i landmark del viso


def create_videos(path, video, where_to_save):
    num_frame = video.split('_')[4]
    frame_contigui = num_frame * 5
    cap= cv2.VideoCapture(path + "/" + videoFile)
    while(cap.isOpened()):
        
        ret, image = cap.read()

        if ret == False:
            break

        rects = detector(image, 1)





path = "/Users/paone/Desktop/prova_video"    #Path della cartella contenente i video da 15 secondi
destinationPath = "/Users/paone/Desktop/csv_computati"  #Path di destinazione per i video csv

for videoFile in os.listdir(path): 
    create_videos(path, videoFile, destinationPath)
