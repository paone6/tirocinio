# -----------------------------------------------------------------------------
# Questo script, data la cartella contenente i file csv, estrae i dati 
# e restituisce il dataset normalizzato
#
# Mario Paone
# ------------------------------------------------------------------------------
import numpy as np
import csv
from sklearn.utils import Bunch
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.datasets import load_iris




path = "D:/prova_pca"  #Path della cartella contenente il dataset
MAX_LINES = 350      #Numero di righe da prendere 
def load_dataset():
    """
    Questo metodo restituisce il dataset normalizzato
    Qui ogni riga del dataset viene normalizzata a parte prima di essere inserita nel dataset
    """
    dataset = []
    targets = []
    for file in os.listdir(path):     #per ogni file video nella cartella
        data = []
        line_count = 1
        with open(path + "/" + file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if(line_count <= MAX_LINES):                #Non prende più di MAX_LINES righe
                    new_row = [float(i) for i in row]    #Converte in float i valori del file csv
                    for element in new_row:
                        data.append(element)            #Aggiunge tutti i valori allo stesso array in modo da avere nel Dataset ogni riga (NumpyArray) rappresenti un file csv     
                    line_count+=1
            data = np.array(data)
            data = preprocessing.MinMaxScaler().fit_transform(data)
            dataset.append(np.array(data))                  
            targets.append(int(file.split('_')[0]))         #Inserisci il primo valore del nome come etichetta
    np_targets = np.array(targets)          #Converte in NumpyArray
    #dataset = preprocessing.normalize(dataset)      #Operazione di normalizzazione
    #dataset = preprocessing.MinMaxScaler().fit_transform(dataset)
    np_dataset = np.array(dataset)      #Converte in NumpyArray
    return Bunch(data = np_dataset, target = np_targets)


a = load_dataset()
print(a.data)