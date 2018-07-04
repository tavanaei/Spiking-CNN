# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 21:29:59 2016

@author: Amir
"""
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import SpikeConv as cnv
with open('Hidden_Weights_RBM_units_128_iteration_6.pickle','rb') as f:
    w=pickle.load(f)['W']

for r in range (20):
    plt.figure()
    for i in range(32):
        plt.subplot(8,4,i+1)
        temp = cnv.Show_Weight(w[i*144:(i+1)*144,r])
        plt.imshow(temp,vmin=0,vmax=1,cmap='Greys_r')
        plt.axis('off')
    plt.show()