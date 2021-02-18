# -----------------------------------------------------------------------------
# Script che si occupa di prendere video da 15 secondi con un viso presente
# e per ogni fotogramma del video, isola le labbra rendendole della stessa 
# dimensione e stampa le distanze euclidee.
# Inoltre stampa una copia del video in cui è visibile solo il riquadro delle
# labbra, ed una copia in cui stampa a video anche i landmark
# Nb. La stampa del video senza landmark è attualmente commentata, alla riga 166
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

SIZE = (300, 200)  #Dimensione a cui verranno stampati i video della labbra 

def save_video(array_video, name, where_to_save):
    """
    Questa funzione prende un array di fotogrammi contigui, il nome del video e il path di destinazione
    e stampa il rispettivo video
    """
    os.chdir(where_to_save)   #cambia il path della destinazione
    print("Produco video " + name)
    framerate = name.split('_')[4]  #salva il framerate del video
    print("Il framerate è: " + str(framerate))
    if(len(array_video) > 0):    #Se l'array non è vuoto
        out = cv2.VideoWriter(str(name) + "_m" + ".avi",cv2.VideoWriter_fourcc(*'DIVX'), int(framerate), SIZE)   #apre il video da salvare
        for frame in array_video:
            out.write(frame)        #aggiunge il frame al video
        out.release()               #rilascia il video e lo salva definitivamente
        print("Completata computazione video "  + str(name))


# Prende una stringa nome ed una matrice di distanze euclidee
# e stampa la matrice 
def print_csv_file(filename, matrix):
    """
    Questa funzione prende una stringa nome ed una matrice
    di distanze euclidee e stampa su file .csv la matrice
    """
    os.chdir(destinationPath) 
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



path = "D:/Video_15_Secondi/da_computare_solo_labbra"    #Path della cartella contenente i video da 15 secondi da computare
destinationPath = "D:/csv/destinazione"  #Path della cartella di destinazione per i video csv
video_destination_path = "D:/Video_15_Secondi/solo_labbra"  #path della cartella di destinazione per i video con presenti solo le labbra
video_destination_path_land = "D:/Video_15_Secondi/destinazione"  #path della cartella di destinazione per i video con presenti i landmark sulle labbra

os.chdir(destinationPath)                             #Cambia la cartella di destinazione in quella di destinationPath

for videoFile in os.listdir(path):     #per ogni file video nella cartella
    print("-----------Inizio computazione " + videoFile + "----------------")
    cap= cv2.VideoCapture(path + "/" + videoFile)       #apre il video
    distanceMatrixExt = []          #Conterrà N array ognuno contenente le distanze euclidee per ogni singolo frame
    video_array = []                #Conterrà i singoli frame delle labbra che verranno usati per formare il video
    video_array_landmark = []       #Conterrà i singoli frame delle labbra con i landmark stampati a video usati per formare il video
    while(cap.isOpened()):          #Fin quando il video non sarà concluso
        ret, image = cap.read()     #Salva ogni frame in image 

        if ret == False:
            break
            
        #image = cv2.resize(image, dsize=(640, 360), interpolation=cv2.INTER_CUBIC)

            
        rects = detector(image, 1)      #Estrae i rettangoli contenenti visi
            

        for rect in rects:      #Per ogni rettangolo contenente un viso
            
            shape = predictor(image, rect)    #Determina i landmark del viso
            shape = shape_to_np(shape)        #Converte i landmark in coordinate (x, y) in un array NumPy  
            
            i = 1
            distanceMatrix = []     #Arrau delle distanze euclidee per i singoli frame
            #print("Stampo robe: " + str(FACIAL_LANDMARKS_IDXS["mouth"][0]) + " " + str(FACIAL_LANDMARKS_IDXS["mouth"][1]))
            (x, y, w, h) = cv2.boundingRect(np.array([shape[FACIAL_LANDMARKS_IDXS["mouth"][0]:FACIAL_LANDMARKS_IDXS["mouth"][1]]])) #Estrae i punti per il rettangolo contenente le labbra
            #print("X = " + str(x) + " Y = " + str(y))
            #print("Larghezza = " + str(w) + "  Altezza = " + str(h))
            roi = image[y-10:y + h +10, x-10:x + w + 10]   #Estrae il rettangolo contenente le labbra(Con dimensione 10 in più da ogni lato)
            new = cv2.resize(roi, dsize=SIZE, interpolation=cv2.INTER_CUBIC)    #Dà al frame dimensione SIZE
            video_array.append(new)     #Aggiunde il frame all'array dei frame del video
            
            xm,ym = FACIAL_LANDMARKS_IDXS["mouth_extern"]  #Prende solo i landmark per le labbra
            new_shape = []
            new_copy = np.copy(new)     #copia l'immagine
            for (xa, ya) in shape[xm:ym]:       #per ogni coppia di coordinate scelta
                #print("("+ str(xa) + ", " + str(ya) + ")")
                xi = xa - (x-10)                                        # In queste 4 righe di codice prende i landmark 
                yi = ya - (y-10)                                        # dell'immagine originale e li scala in modo da 
                new_x = int((xi * SIZE[0]) / (w+20))                    # adattarli alle nuove dimensioni scandite dalla
                new_y = int((yi * SIZE[1]) / (h+20))                    # costante SIZE
                #print("x = " + str(new_x) + " y = " + str(new_y))
                cv2.circle(new_copy, (new_x, new_y), 1, (0,0,255), -1)  #Stampa i landmark sull'immagine
                new_shape.append((new_x, new_y))                        #Aggiunte le coordiante dei frame all'array
            video_array_landmark.append(new_copy)                       #Aggiunge le immagini all'array per stampare il video
            #cv2.imshow("prova",new)
            #cv2.waitKey()


            #xm,ym = FACIAL_LANDMARKS_IDXS["mouth_intern"]  #Prende solo i landmark per le labbra
            i=1
            for (x1, y1) in new_shape:
                #cv2.circle(image, (x, y), 1, (0, 0, 255), -1)  #Disegna i landmark sull'immagine
                #print("("+ str(x1) + ", " + str(y1) + ")")
                for (x2, y2) in new_shape[i:]:
                    #print("("+ str(x2) + ", " + str(y2) + ")")
                    distanceMatrix.append(int(np.linalg.norm(np.array([x1,y1]) - np.array([x2,y2]))))   #Stampa le distanze
                i+=1
            distanceMatrixExt.append(distanceMatrix)
            
    save_video(video_array, videoFile.split('.')[0],video_destination_path)            #Chiama la funzione per stampare il video della labbra senza landmark
    save_video(video_array_landmark, videoFile.split('.')[0],video_destination_path_land)     #Chiama la funzione per stampare il video della labbra con i landmark                 
    print_csv_file(videoFile.split(".")[0] + "_m", distanceMatrixExt)   #Chiama la funzione per stampare la matrice di distanze nell'omonimo file csv       
    print("-----------Conclusa computazione " + videoFile + "----------------")
  
    
cap.release()
cv2.destroyAllWindows()