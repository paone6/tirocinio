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
MAX_LINES = 350
def load_dataset():
    """
    Questo metodo restituisce il dataset normalizzato 
    Qui ogni file csv viene normalizzato interamente all'inizio 
    """
    dataset = []
    targets = []
    for file in os.listdir(path):     #per ogni file video nella cartella
        data = []
        line_count = 1
        df=pd.read_csv(path + "/" + file, sep=',',header=None)
        df = preprocessing.MinMaxScaler().fit_transform(df)    #Normalizza i valori del file
        #df = preprocessing.normalize(df, norm='l2')
        for row in df:
                if(line_count <= MAX_LINES):      #Non prende più di MAX_LINES righe
                    new_row = [float(i) for i in row]   #Converte in flaot i valori del file csv
                    for element in new_row:
                        data.append(element)          #Aggiunge tutti i valori allo stesso array in modo da avere nel Dataset ogni riga (NumpyArray) rappresenti un file csv
                    line_count+=1
        dataset.append(np.array(data))                  
        targets.append(int(file.split('_')[0]))      #Inserisci il primo valore del nome come etichetta
    np_targets = np.array(targets)      #Converte in NumpyArray
    #dataset = preprocessing.normalize(dataset)
    #dataset = preprocessing.MinMaxScaler().fit_transform(dataset)
    np_dataset = np.array(dataset)    #Converte in NumpyArray
    return Bunch(data = np_dataset, target = np_targets)


a = load_dataset()
print(a.data.shape)
b = np.transpose(a.data)
print(b.shape)