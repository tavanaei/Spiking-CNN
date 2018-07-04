# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:03:43 2017

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
D1 = 32
D2 = 64
units = 64
W=np.random.random([(patch_size**2)*D2,units])

wf_1 = open('Filters/'+str(D1)+'/filter_1_U'+str(D1)+'_P5_MNIST_lagrange_thDynamic_10000.pickle','rb')
w_1 = pickle.load(wf_1)['W']

wf_2 = open('Filters/D2/filter_2_U'+str(D2)+'_D'+str(D1)+'_P5_MNIST_lagrange_thDynamic_5000.pickle','rb')
w_2 = pickle.load(wf_2)['W']

threshold = 0.9
threshold_2 = 0.7
thr=.35
thr_2=.3
save_index = 0.0
pool_stride = 2
for iter in range(1):
    print(iter)
    for r in range(R):
        if (np.log10((r+1))==save_index or (r+1)%1000==0):
            
            print(str(r+1)+" Images Are Done")
            
            with open('Filters/filter_2_U'+str(units)+'_D'+str(D1)+'_'+str(D2)+'_P5_MNIST_lagrange_thDynamic_'+str(r+1)+'.pickle','wb') as f2:
                save={'W':W}
                pickle.dump(save,f2)
            save_index+=1
        digit = cnv.Image_Read(data[r,:],28)
        spike_digit = cnv.Poisson_2D(digit,T)
        fmaps = cnv.Reconstruct(w_1, spike_digit, patch_size, threshold, T,True,True)
        _,pmaps = cnv.MaxPool(fmaps,pool_stride)
        fmaps2 = cnv.Reconstruct_Hidden(pmaps,patch_size,D2,w_2,threshold_2)
        _,pmaps2 = cnv.MaxPool(fmaps2,pool_stride)
        
        W,thr = cnv.Autoencoder_Hidden_Train(pmaps2,W,patch_size,units,thr_2,2.0)
        
plt.figure()
for i in range(units):
    plt.subplot(4,4,i+1)
    plt.imshow(cnv.Show_Weight(W[:,i]),cmap='Greys_r')
    plt.axis('off')
plt.show()
