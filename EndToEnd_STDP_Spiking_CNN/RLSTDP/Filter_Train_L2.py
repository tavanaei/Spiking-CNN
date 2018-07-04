# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:03:43 2017

@author: Amir
"""

import numpy as np
from six.moves import cPickle as pickle
import SpikeConv as cnv
import matplotlib.pyplot as plt

data,label = cnv.Read_Data('Data.pickle') # for MNIST it is True and output is data,label

R=5000
T=40
patch_size = 5
D1 = 16
units = 16
W=np.random.random([(patch_size**2)*D1,units])

wf = open('Filters/'+str(D1)+'/filter_1_U'+str(D1)+'_P5_MNIST_lagrange_thDynamic_10000.pickle','rb')
w_1 = pickle.load(wf)['W']

threshold = 0.8
thr=.35
save_index = 0.0
pool_stride = 2
for iter in range(1):
    print(iter)
    for r in range(R):
        if (np.log10((r+1))==save_index or (r+1)%1000==0):
            
            print(str(r+1)+" Images Are Done")
            
            with open('Filters/filter_2_U'+str(units)+'_D'+str(D1)+'_P5_MNIST_lagrange_thDynamic_'+str(r+1)+'.pickle','wb') as f2:
                save={'W':W}
                pickle.dump(save,f2)
            save_index+=1
        digit = cnv.Image_Read(data[r,:],28)
        spike_digit = cnv.Poisson_2D(digit,T)
        fmaps = cnv.Reconstruct(w_1, spike_digit, patch_size, threshold, T,True,True)
        _,pmaps = cnv.MaxPool(fmaps,pool_stride)
        
        W,thr = cnv.Autoencoder_Hidden_Train(pmaps,W,patch_size,units,thr)
        
plt.figure()
for i in range(units):
    plt.subplot(4,units/4,i+1)
    plt.imshow(cnv.Show_Weight(W[:,i]),cmap='Greys_r')
    plt.axis('off')
plt.show()
