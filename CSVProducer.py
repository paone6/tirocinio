# -----------------------------------------------------------------------------
# Script che si occupa di prendere video da 15 secondi con un viso presente
# e per ogni video produce un file csv contenente le distanze euclidee 
# tra i landmark della bocca
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



# Definisce un dizionario che mappa gli indici per i
# landmark facciali per ogni regione del viso
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
    ("mouth_intern", (60, 68)),
    ("mouth_extern", (48, 60)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

# Prende una stringa nome ed una matrice di distanze uclidee
# e stampa la matrice 
def print_csv_file(filename, matrix):
    """
    Questa funzione prende una stringa nome ed una matrice
    di distanze euclidee e stampa su file .csv la matrice
    """
    with open(str(filename)+".csv", mode='w', newline='') as csv_file:  #apre il file csv
        writer = csv.writer(csv_file)
        #title = list(range(1,191))
        #writerow(title)
        for lista in matrix:    #per ogni frame del video
            writer.writerow(lista)



def shape_to_np(shape, dtype="int"):
    """
    Restituisce le una lista contente le coppie
    di coordinate che rappresentano i landmark
    """
    
    coords = np.zeros((68, 2), dtype=dtype)  #inizializza la lista
    
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


detector = dlib.get_frontal_face_detector()  #inizializza il face detector(HOG-based) della libreria dlib
predictor = dlib.shape_predictor("/Users/paone/Desktop/VideoDataset/shape_predictor_68_face_landmarks.dat")  #crea il predictor per i landmark del viso


path = "D:/Video_15_Secondi/senza_csv_8_land"    #Path della cartella contenente i video da 15 secondi
destinationPath = "D:/csv_video_15_secondi_8_landmark"  #Path di destinazione per i video csv
os.chdir(destinationPath)                             #Cambia la cartella di destinazione in quella di destinationPath

for videoFile in os.listdir(path):     #per ogni file video nella cartella
    print("-----------Inizio computazione " + videoFile + "----------------")
    cap= cv2.VideoCapture(path + "/" + videoFile)
    distanceMatrixExt = []
    while(cap.isOpened()):
        ret, image = cap.read()

        if ret == False:
            break
            
        image = cv2.resize(image, dsize=(640, 360), interpolation=cv2.INTER_CUBIC)

            
        rects = detector(image, 1)
            

        for rect in rects:
            
            shape = predictor(image, rect)    #Determina i landmark del viso
            shape = shape_to_np(shape)        #Converte i landmark in coordinate (x, y) in un array NumPy  
                
            i = 1
            distanceMatrix = []

            xm,ym = FACIAL_LANDMARKS_IDXS["mouth_intern"]  #Prende solo i landmark per le labbra
            for (x, y) in shape[xm:ym]:
                #cv2.circle(image, (x, y), 1, (0, 0, 255), -1)  #Disegna i landmark sull'immagine
                for (x2, y2) in shape[xm+i:ym]:
                    distanceMatrix.append(np.linalg.norm(np.array(x,y) - np.array(x2,y2)))
                i+=1
            distanceMatrixExt.append(distanceMatrix)
            #cv2.imshow('Window', image)
            #cv2.waitKey()
            
            
    print_csv_file(videoFile.split(".")[0], distanceMatrixExt)   #Chiama la funzione per stampare la matrice di distanze nell'omonimo file csv       
    print("-----------Conclusa computazione " + videoFile + "----------------")
  
    
cap.release()
cv2.destroyAllWindows()