# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 17:11:39 2016

@author: Amir
"""


import matplotlib.pyplot as plt
import numpy as np
import SpikeConv as cnv
import cPickle as pickle
import multiprocessing as mp
import os

R=3000
depth = 32
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

Potential = np.zeros([R*20,hiddens])
Post_Spikes = np.zeros([R,hiddens,T])

weight_name = "Hidden_Weights_RBM_units_128_iteration_6"
   
with open(weight_name+".pickle",'rb') as f2:
    weights = pickle.load(f2)
    w_hidden = weights['W']
    
#w_hidden = cnv.Normalize(w_hidden)

print("****** Data preparation is done ********")

def CSV_Write(data,label):
   f_id = open(weight_name+"_FinalTrain.csv",'w')
   for i in range(hiddens):
       f_id.write('V'+str(i)+',')
   f_id.write('Label')
   f_id.write('\n')
   for i in range(0,R*20):
       for j in range(hiddens):
           f_id.write(str(data[i,j])+',')
       f_id.write(str(int(label[i])))
       f_id.write('\n')
   f_id.close()

def log_result_hidden(result):
    #print(result[0])
    Potential[result[3]*3000+result[2],:]=result[0]
    Post_Spikes[result[2],:,:]=result[1]

def Partial_Convolution(pre_spikes,index,part):
    threshold = 0.06
    u_tot,post_spike = cnv.Hidden_NN3(pre_spikes,w_hidden,threshold,hiddens)    
    return u_tot,post_spike,index,part

def Hidden_NN(part,preSpikes):
    if __name__ == '__main__':
        ##start_time = time.time()
        pool = mp.Pool()    
        for i in range(R):
            pool.apply_async(Partial_Convolution, args = (preSpikes[i,:,:],i,part), callback = log_result_hidden)
        pool.close()
        pool.join()
        
        ##end_time = time.time()
        print("*************** Hidden Calculations are Done! ****************")


for part in range (11):
    if (part==10):
        ddd,label = cnv.Read_Data("Data_Test_MNIST.pickle")
        CSV_Write(Potential,label) #Potential    sum(preSpikes[:,:,:].T).T
        continue
    print ("Part: "+str(part+1)+" is running")
    preSpikes = np.zeros([R,afferents,T])
    files = os.listdir("CompleteData/p"+str(part+1))
    files.sort()
    for i in range(sec):
        print(files[i])
        preSpike_sec = Load("CompleteData/p"+str(part+1)+"/"+files[i]) 
        preSpikes[i*100:(i+1)*100,:,:]=Flat(preSpike_sec)
        del preSpike_sec
    Hidden_NN(part,preSpikes)




