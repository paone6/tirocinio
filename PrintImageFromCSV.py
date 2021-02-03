# -----------------------------------------------------------------------------
# Questo script, dato un file csv in input, normalizza i suoi valori in
# in valori tra 0 e 255 e stampa l'immagine corrispondente
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
from PIL import Image


path = "/Users/paone/Desktop/csv_computati"    #Path della cartella contenente i video da 15 secondi
destinationPath = "/Users/paone/Desktop/csv_computati"  #Path di destinazione per i video csv
os.chdir(destinationPath)   

def print_image(array, nome):
    img_w, img_h = len(array[0]), len(array)
    print(str(img_w) + " " + str(img_h))
    #data = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    #data[100, 100] = [255, 0, 0]
    for row in array:
        print(row)
    img = Image.fromarray(array,'L')
    img.save(nome.split('.')[0] + ".png")
 

for csvFile in os.listdir(path): 
    print("Aperto file " + csvFile)
    normalized_matrix = []
    with open(path + "/" + csvFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if(line_count < 360):
                line_count+=1
                float_row = [float(i) for i in row]
                normalized_row =  np.array(255*(float_row - np.min(float_row))/np.ptp(float_row)).astype(int) 
                normalized_matrix.append(normalized_row)
        np_matrix = np.array(normalized_matrix)        
    print_image(np_matrix, csvFile)
                
            

    