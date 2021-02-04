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

CONTIGUOUS_SECONDS = 5

def save_video(contiguous_frames, where_to_save, framerate, name):
    
    os.chdir(where_to_save)                             #Cambia la cartella di destinazione in quella di destinationPath

    #img = cv2.imread(contiguous_frames[0][0])
    height, width= len(contiguous_frames[0][0]), len(contiguous_frames[0][0][0])
    size = (width,height)
    print("Dimensione video = " + str(size))
    print("Numero di frame al secondo = " + str(framerate))
    label = 1
    if(len(contiguous_frames) >=3):
        for i in range(len(contiguous_frames) // 3):
            out = cv2.VideoWriter(str(name) + "_" + str(label) + ".avi",cv2.VideoWriter_fourcc(*'DIVX'), int(framerate), size)
    

            for j in range(i*3,i*3+3):
                for image in contiguous_frames[j]:
                    out.write(image)
            out.release()   
            print("Completata computazione video: " + str(name) + "_" + str(label))
            label+=1


def create_videos(path, video, where_to_save):
    print("Apertura file " + video)
    num_frame = video.split('.')[0].split('_')[4]
    print("Num frame = " + str(num_frame))
    frame_contigui = int(num_frame) * CONTIGUOUS_SECONDS
    print("Grandezza sequenze = " + str(frame_contigui))
    cap= cv2.VideoCapture(path + "/" + videoFile)
    contiguous_sequences = []  #contiene array di 5 secondi di sequenze ciascuno 
    contiguous_frames = []     #contiene sequenze continue di frame di lunghezza massima 5 secondi

    while(cap.isOpened()):
        print(str(len(contiguous_frames)))
        if(len(contiguous_frames) == frame_contigui):
            print("Trovata sequenza di " + str(CONTIGUOUS_SECONDS) + " secondi")
            contiguous_sequences.append(contiguous_frames)
            contiguous_frames = []
        
        ret, image = cap.read()

        if ret == False:
            break

        #cv2.imshow('image',image)
        #cv2.waitKey(0)
        rects = detector(image, 1)
        if(len(rects) == 1):
            print("Aggiunto frame")
            contiguous_frames.append(image)
        else:
            print("Frame senza un soggetto visibile, sequenza cancellata")
            contiguous_frames = []

    save_video(contiguous_sequences, where_to_save, num_frame, video.split('.')[0])       






path = "/Users/paone/Desktop/prova_video"    #Path della cartella contenente i video da 15 secondi
destinationPath = "/Users/paone/Desktop/csv_computati"  #Path di destinazione per i video csv

for videoFile in os.listdir(path): 
    create_videos(path, videoFile, destinationPath)
