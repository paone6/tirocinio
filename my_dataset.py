import numpy as np
import csv
from sklearn.utils import Bunch
import pandas as pd
import os


path = "D:/prova_pca"  #Path della cartella contenente il dataset
MAX_LINES = 350
def load_dataset():
    dataset = []
    targets = []
    for file in os.listdir(path):     #per ogni file video nella cartella
        data = []
        line_count = 1
        with open(path + "/" + file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if(line_count <= MAX_LINES):
                    new_row = [float(i) for i in row]
                    for element in new_row:
                        data.append(element)
                    line_count+=1
            dataset.append(np.array(data))                  
            targets.append(int(file.split('_')[0]))
    np_targets = np.array(targets)
    np_dataset = np.array(dataset, dtype=object)
    return Bunch(data = np_dataset, target = np_targets)



