# -----------------------------------------------------------------------------
# Script che prende il dataset dalla cartella contenente i file csv,
# e utilizzando PCA diminuisce il numero di colonne per ogni file
#
# Mario Paone
# ------------------------------------------------------------------------------

import numpy as np
import csv
from sklearn.utils import Bunch
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


NUM_COMPONENTI = 11         #Num di colonne a cui diminuire tramite PCA
NUM_FRAMES = 350            #Numero massimo di righe da prendere

path = "D:/prova_pca"  #Path della cartella contenente il dataset
def load_dataset():
    datas = []
    targets = []
    for file in os.listdir(path):
        data = []
        line_count = 1
        names = [str(i) for i in range(1,67)]
        dataset = pd.read_csv(path + "/" + str(file),names = names)
        sc = StandardScaler()
        dataset = sc.fit_transform(dataset)
        pca = PCA(n_components= NUM_COMPONENTI)
        dataset = pca.fit_transform(dataset)
        for row in dataset:
            if(line_count <= NUM_FRAMES):
                for element in row:
                    data.append(element)
                line_count+=1
        targets.append(int(str(file).split('.')[0].split('_')[0]))  
        datas.append(np.array(data))  
    np_targets = np.array(targets)
    np_dataset = np.array(datas)
    return Bunch(data = np_dataset, target = np_targets)
        
       

        
      

    