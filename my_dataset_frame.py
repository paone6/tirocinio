# -----------------------------------------------------------------------------
# Script che prende il dataset dalla cartella contenente i file csv,
# setta correttamente i dati li restituisce sotto forma di Bunch contenente
# data e target
#
# Mario Paone
# ------------------------------------------------------------------------------

import numpy as np
import csv
from sklearn.utils import Bunch
import pandas as pd
import os


path = "D:/csv_video_15_secondi"  #Path della cartella contenente il dataset

def load_my_dataset():
    """
    Questa funzione legge i file csv e mette i valori in numpy array e restituisce dati e target 
    settati correttamente
    """
    dataset = []
    targets = []
    for file in os.listdir(path):     #per ogni file video nella cartella
        with open(path + "/" + file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            file_distances = []
            line_count = 0
            for row in csv_reader:
                if(line_count < 360):
                    new_row = [float(i) for i in row]
                    file_distances += new_row
                    line_count += 1
                #dataset.append(np_new_row)               
            targets.append(int(file.split('_')[0]))
            dataset.append(np.array(file_distances))
    np_targets = np.array(targets)
    np_dataset = np.array(dataset)
    return Bunch(data = np_dataset, target = np_targets)



