# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 09:40:46 2017

@author: Amir
"""
from RLSTDP import SpikeConv as cnv
import numpy as np
from six.moves import cPickle as pickle
import multiprocessing as mp
#import matplotlib.pyplot as plt
import time

R=35000
np.random.seed(0)
with open('Data_Train_MNIST.pickle','rb') as f:
    save = pickle.load(f)
    dataA=save['Data']
    labelA=save['Label']
 
with open('Data_Test_MNIST.pickle','rb') as f2:
    save2 = pickle.load(f2)
    dataB=save2['Data']
    labelB=save2['Label'].reshape([10000])

fullData = np.zeros([70000,785])
fullData[:60000,:784] = dataA
fullData[:60000,784] = labelA.squeeze()
fullData[60000:,:784] = dataB
fullData[60000:,784] = labelB.squeeze()
np.random.shuffle(fullData)

T=20
patch_size = 5
threshold=0.8 #0.8 for RL we chose 0.4
pool_stride = 2

grow = [2,2]
unit_set = [16,32,16]
NoConv=2

path = 'Filters/'
w_names = [str(unit_set[0])+'/filter_1_U'+str(unit_set[0])+'_P5_MNIST_lagrange_thDynamic_10000.pickle',
'D2/filter_2_U'+str(unit_set[1])+'_D'+str(unit_set[0])+'_P5_MNIST_lagrange_thDynamic_5000.pickle',
'D3/filter_2_U'+str(unit_set[2])+'_D'+str(unit_set[0])+'_'+str(unit_set[1])+'_P5_MNIST_lagrange_thDynamic_5000.pickle']

Ws=list()
Ds = "RL8_Maps_"

#def show_map(p_maps,level):
#    dd = len(p_maps)
#    plt.figure()
#    for i in range(dd):
#        plt.subplot(8,dd/4,i+1)
#        plt.imshow(sum(p_maps[i].T).T,vmax = 38, vmin = 0)
#        plt.title(str(level))
#        plt.axis('off')
#    plt.show()

print('Read weights man!')
for w_l in range(NoConv):
    with open(path+w_names[w_l],'rb') as f_w:
        Ws.append(pickle.load(f_w)['W'])
    Ds= Ds+str(unit_set[w_l])



def FeatureExtract(spike_digit,Ws,patch_size,threshold,T,pool_stride):
    Layer = 0
    map_test = list()
    for w in Ws:
        if (Layer==0):
            _,pmaps,mapTemp = cnv.MaxPool(cnv.Reconstruct(w, spike_digit, patch_size, threshold, T,True,True),pool_stride)
        else:        
            _,pmaps,mapTemp = cnv.MaxPool(cnv.Reconstruct_Hidden(pmaps,patch_size,np.shape(w)[1],w,threshold),pool_stride)
        #show_map(pmaps,Layer)
        map_test.append(mapTemp)
        Layer+=1
        threshold = threshold-.15#*2
    return map_test#pmaps

print('Generate spike trains!')
FeatureMaps = [None]*R
spike_digit=list()

def log_result_hidden(result):
    print(result[1])
    FeatureMaps[result[1]]=result[0]

def Partial_Convolution(vec,index):
    digit = cnv.Image_Read(vec,28)
    pre_spikes = cnv.Poisson_2D(digit,T)
    mapy = FeatureExtract(pre_spikes,Ws,patch_size,threshold,T,pool_stride)    
    return mapy,index

def FeatureCNN():
    if __name__ == '__main__':
        start_time = time.time()
        pool = mp.Pool()    
        for i in range(R):
            pool.apply_async(Partial_Convolution, args = (fullData[i,:784],i), callback = log_result_hidden)
        pool.close()
        pool.join()
        
        end_time = time.time()
        print(end_time-start_time)
        
print('Lets Dance!')        
FeatureCNN()
print("*************** Calculations are Done! ****************")

#for i in range(100):
#    print(i)
#    digit = cnv.Image_Read(fullData[i,:784],28)
#    pre_spikes = cnv.Poisson_2D(digit,T)
#    mapy = FeatureExtract(pre_spikes,Ws,patch_size,threshold,T,pool_stride)


print('save results')
with open(Ds+'.pickle','wb') as fSave:
    save = {'Maps':FeatureMaps,
            'Labels':fullData[:,784]}
    pickle.dump(save,fSave)



## TEST
#plt.figure()
#for i in range(16):
#    plt.subplot(4,4,i+1)
#    plt.axis('off')
#    plt.imshow(cnv.Show_Weight(Ws[0][:,i]),cmap='Greys_r')