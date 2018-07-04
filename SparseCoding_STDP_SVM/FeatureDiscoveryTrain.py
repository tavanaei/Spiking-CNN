# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:49:07 2016

@author: Amir
"""

#import matplotlib.pyplot as plt
import numpy as np
import SpikeConv as cnv
from six.moves import cPickle as pickle
import os

depth = 16

R=3000
afferents = depth*12*12
hiddens=64
T=20
def Load(file_name):
    with open(file_name,'rb') as f:
        save = pickle.load(f)
    return save['pooled_maps']
        
def Flat(data):
    r=np.shape(data)[0]
    d=np.shape(data)[1]
    p1=np.shape(data)[2]
    p2=np.shape(data)[3]
    t=np.shape(data)[4]
        
    temp = np.zeros([r,d*p1*p2,t])
        
    for row in range(r):
        for depth in range(d):
            for pixel1 in range(p1):
                for pixel2 in range(p2):
                    temp[row,(depth*p1*p2)+(pixel1*p2)+pixel2,:]=data[row,depth,pixel1,pixel2,:]
    return temp

sec = int(R/100)
w=np.random.random([afferents,hiddens])
threshold=0.5
for iteration in range(6):
    print("######  Iteration: "+str(iteration+1)+"  ######")
    for part in range(5):
        files = os.listdir("p"+str(part+1))
        preSpikes = np.zeros([R,afferents,T])
        print("*****  Loading 3000 Data in section: "+str(part+1))
        i=0
        for file_pool in files:
            print(file_pool)
            if (file_pool=='.DS_Store'):
                continue
            preSpike_sec = Load("p"+str(part+1)+"/"+file_pool) 
            preSpikes[i*100:(i+1)*100,:,:]=Flat(preSpike_sec)
            i=i+1
            del preSpike_sec
            
        print("****** Data preparation is done ********")

        w=cnv.NN_Layer_2(preSpikes,T,w,threshold,hiddens,cnv.STDPProb)

        if (iteration==0 and part<4):
            with open("Hidden_Weights_RBM_units_"+str(hiddens)+"_iteration_"+str(iteration)+"_"+str(part+1)+".pickle",'wb') as f0:
                save0 = {'W': w}
                pickle.dump(save0,f0,pickle.HIGHEST_PROTOCOL)
        
    with open("Hidden_Weights_RBM_units_"+str(hiddens)+"_iteration_"+str(iteration+1)+".pickle",'wb') as f2:
        save = {'W': w}
        pickle.dump(save,f2,pickle.HIGHEST_PROTOCOL)
    
#
#plt.figure()
#for i in range(hiddens):
#    plt.subplot(10,10,i+1)
#    temp=cnv.Show_Weight(w[:,i])
#    plt.imshow(temp,cmap='Greys_r')
#    plt.axis('off')
#plt.show()

##for r in range (20):
##    plt.figure()
##    for i in range(64):
##        plt.subplot(8,8,i+1)
##        temp = cnv.Show_Weight(w[i*144:(i+1)*144,r])
##        plt.imshow(temp,vmin=0,vmax=1,cmap='Greys_r')
##        plt.axis('off')
##    plt.show()
##    
##    
               
