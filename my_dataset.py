import numpy as np
import csv
from traitlets.utils.bunch import Bunch
import pandas as pd
import os


path = "/Users/paone/Desktop/csv_computati"  #Path della cartella contenente il dataset

def load_my_dataset():
    dataset = []
    targets = []
    for file in os.listdir(path):     #per ogni file video nella cartella
        matrix = []
        with open(path + "/" + file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                new_row = [float(i) for i in row]
                matrix.append(new_row)
            targets.append(int(file.split('_')[0]))
            dataset.append(matrix)
    return Bunch(data = dataset, target = targets)

b = load_my_dataset()
print(b.data)
print(len(b.data[0]))
print(len(b.data[1]))
print(len(b.data[2]))
print(len(b.data[3]))

print(len(b.data))
print(b.target)




