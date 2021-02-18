# -----------------------------------------------------------------------------
# Questo script, dato un video in input, si occuppa di estrarre un numero
# prefissato di sottovideo di 15 secondi in cui il volto del soggetto
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

CONTIGUOUS_SECONDS = 5 #numero di secondi consecutivi necessari
MAX_PER_VIDEO = 5       #numero di video massimo da produrre

def save_video(contiguous_frames, where_to_save, framerate, name):
    """
    Questa funzione prende una array contenente i frame contigui, una catella di destionazione, il FrameRate 
    e il nome del video e salva il numero massimo di sottovideo computabili tramite i frame contenuti in contigous_frame
    e salva i video prodotti nella cartella where_to_save
    """
    
    os.chdir(where_to_save)                             #Cambia la cartella di destinazione in quella di destinationPath

    if(len(contiguous_frames) > 0):     #L'array deve contenere almeno una sequenza di 5 secondi 
        

        #img = cv2.imread(contiguous_frames[0][0])
        height, width= len(contiguous_frames[0][0]), len(contiguous_frames[0][0][0])    #Prende altezza e larghezza del video
        size = (width,height)               #Setta la dimensione del video
        print("Dimensione video = " + str(size))
        print("Numero di frame al secondo = " + str(framerate))
        label = 1           #Numeri sequenziali per etichettari i video prodotti
        if(len(contiguous_frames) >=3):             #Se contigous_frame contiene almeno 3 sequenze da TOT secondi ciascuna
            for i in range(len(contiguous_frames) // 3):        
                out = cv2.VideoWriter(str(name) + "_" + str(label) + ".avi",cv2.VideoWriter_fourcc(*'DIVX'), int(framerate), size)   #crea il video
        

                for j in range(i*3,i*3+3):
                    for image in contiguous_frames[j]:
                        out.write(image)
                out.release()   
                print("Completata computazione video: " + str(name) + "_" + str(label))
                label+=1


def create_videos(path, video, where_to_save):
    """
    Questa funzione analizza ogni frame di un video e salva in un array tutte le sequenze di frame contigue e chiama il metodo save_video
    per salvare i video computabili da quelle sequenze
    """
    print("Apertura file " + video)
    num_frame = video.split('.')[0].split('_')[4]           #Salva il frameRate
    print("Num frame = " + str(num_frame))
    frame_contigui = int(num_frame) * CONTIGUOUS_SECONDS        #Calcola la grandezza delle sequenze
    print("Grandezza sequenze = " + str(frame_contigui))
    cap= cv2.VideoCapture(path + "/" + videoFile)       #Apre il video
    contiguous_sequences = []  #contiene array di sequenze da 5 secondi ciascuno 
    contiguous_frames = []     #contiene sequenze continue di frame di lunghezza massima 5 secondi

    while(cap.isOpened()):
        print(str(len(contiguous_frames)))
        if(len(contiguous_frames) == frame_contigui):           #Quando la sequenza raggiunge la grandezza massima la aggiunge all'array e va avanti cercando altre sequenze 
            print("Trovata sequenza di " + str(CONTIGUOUS_SECONDS) + " secondi")
            contiguous_sequences.append(contiguous_frames)          #Aggiunge la sequenza trova all'array
            print("Sono presenti " + str(len(contiguous_sequences)) + " sequenze nell'array")
            contiguous_frames = []                  #Svuola l'array per cercare nuove sequenze
        
        ret, image_o = cap.read()       #Apre il frame
        image = cv2.resize(image_o, dsize=(640, 360), interpolation=cv2.INTER_CUBIC)        #Cambia la dimensione del frame in modo che ogni video abbia la stessa dimensione
        if ret == False:
            break

        #cv2.imshow('before',image_o)
        #cv2.imshow('after',image)
        #cv2.waitKey(0)
        rects = detector(image, 1)      #Trova il viso nell'immagine
        if(len(rects) == 1):
            print("Aggiunto frame")
            contiguous_frames.append(image)         #Se il frame contiene un viso lo aggiunge
        else:
            print("Frame senza un soggetto visibile, sequenza cancellata")
            contiguous_frames = []      #Se il frame non contiene un viso allora cancella l'intera sequenza

        if(len(contiguous_sequences) >= (MAX_PER_VIDEO * 3)):       #Ha raggiunto il numero massimo di sequenze
            print("Prodotti " + str(MAX_PER_VIDEO) + " video da questo, proseguo col prossimo")     
            break

    save_video(contiguous_sequences, where_to_save, num_frame, video.split('.')[0])       #Chiama il metodo save_video e gli passa l'array di sequenze trovate






path = "D:/da_computare"    #Path della cartella contenente i video integrali
destinationPath = "D:/Video_15_Secondi/destinazione"  #Path di destinazione per i video da 15 secondi

for videoFile in os.listdir(path): 
    create_videos(path, videoFile, destinationPath)
