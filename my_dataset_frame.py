import numpy as np
import csv
from sklearn.utils import Bunch
import pandas as pd
import os


path = "/Users/paone/Desktop/csv_computati"  #Path della cartella contenente il dataset

def load_my_dataset():
    dataset = []
    targets = []
    for file in os.listdir(path):     #per ogni file video nella cartella
        with open(path + "/" + file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                new_row = [float(i) for i in row]
                np_new_row = np.array(new_row)
                dataset.append(np_new_row)               
                targets.append(int(file.split('_')[0]))
    np_targets = np.array(targets)
    np_dataset = np.array(dataset, dtype=object)
    return Bunch(data = np_dataset, target = np_targets)





