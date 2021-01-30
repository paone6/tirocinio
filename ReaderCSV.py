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


def read_csv(name, path):
    matrix = []
    with open(path + "/" + name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            new_row = [float(i) for i in row]
            matrix.append(new_row)
    return matrix

        

sourcePath = "/Users/paone/Desktop/csv_computati"  
for csvFile in os.listdir(sourcePath): 
    matr = read_csv(csvFile, sourcePath)




