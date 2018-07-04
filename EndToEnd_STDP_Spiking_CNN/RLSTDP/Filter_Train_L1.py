# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:54:58 2016

@author: Amir
"""

import numpy as np
from six.moves import cPickle as pickle
import SpikeConv as cnv
import matplotlib.pyplot as plt

data,label = cnv.Read_Data('Data.pickle',True) # for MNIST it is True and output is data,label

R=15000 # for MNIST it is  10000   5
T=40
patch_size = 5 # for MNIST 5  Nature 16
units =32 #256#64
W=np.random.random([patch_size**2,units])
#W=np.ones([patch_size**2,units])*.5

thr=.15 # MNIST  Nature .25
save_index = 0.0
landa = 0
tau=0.25#5
epsilon = 0#9
inhibition = 0#0.4

for iter in range(1): # large:2
    print(iter)
    for r in range(R):
        #print(r+1)
        if (r==0 or r==9 or r==99 or (r+1)%1000.0==0): #(r==4 or r==0):#:##
            
            print(str(r+1)+" Images Are Done")
            #s=''
#            if ((r+1)/100<10):
#                s = '0'
            with open('Filters/filter_1_U'+str(units)+'_P5_MNIST_lagrange_thDynamic_T40_Lambda'+str(landa)+'_Tau'+str(tau)+'_Inh'+str(inhibition)+'_'+str(int(r+1))+'.pickle','wb') as f2:
                save={'W':W}
                pickle.dump(save,f2)
            save_index+=1
#            plt.figure()
#            for i in range(units):
#                plt.subplot(32,16,i+1)
#                plt.axis('off')
#                plt.imshow(cnv.Show_Weight(W[:,i]),cmap='Greys_r')
        digit = cnv.Image_Read(data[r,:],28) # for MNIST is 28 # 1- .... 512
        #digit[digit<=0.2]=0
        #digit[digit>=0.8]=1
        spike_digit = cnv.Poisson_2D(digit,T) # digit*255  #### cnv.Poisson_2D(digit,T)
        W,thr = cnv.Autoencoder_L1_Train_Thr(spike_digit,units,patch_size,W,thr,landa,tau,epsilon,inhibition)
        



#with open('filter_1_U8_P5_MNIST_prob_thDynamic.pickle','wb') as f2:
#    save={'W':W}
#    pickle.dump(save,f2)
    
            