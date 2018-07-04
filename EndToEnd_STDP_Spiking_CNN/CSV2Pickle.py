# -*- coding: utf-8 -*-
"""
Created on Tue May 24 12:10:31 2016

Feature Extraction , MNIST

@author: Amir
"""
import numpy as np
from six.moves import cPickle as pickle

dataFile = open("train.csv",'r')

index = 0
R=3000
#dataFile.readline()
label = np.zeros([R,1])
data = np.zeros([R,784])
index = 0
while(dataFile):
    if (index%100==0):
        print(index)
    line = dataFile.readline()
    pixTemp = line.split(',')
    if (index==R-1):
        print(line)
    try:
        label[index]=int(pixTemp[0])
        data[index,:] = [int(x) for x in pixTemp[1:785]]
        index+=1
    except:
        print("End of File")
        break

with open("MNIST_Train.pickle",'wb') as f:
    save = {'Data': data,
            'Label': label}
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            
print("DONE")
