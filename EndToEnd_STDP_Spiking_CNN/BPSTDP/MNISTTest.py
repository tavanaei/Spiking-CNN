# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:04:50 2017

@author: amir
"""

import numpy as np
import SVP as svp
import matplotlib.pylab as plt
import cPickle as pickle

with open('Data_Train_MNIST.pickle','rb') as f:
    save = pickle.load(f)
    data=save['Data']/255.0
    label=save['Label']
 
with open('Data_Test_MNIST.pickle','rb') as f2:
    save2 = pickle.load(f2)
    dataTest=save2['Data']/255.0
    labelTest=save2['Label'].reshape([10000])
    
T=50
hiddenThr = [10,10]
outputThr = 20#[10,15,20,30,50,75] #50 #15 #Large threshold for 1000 and 1500 H
tau = 1

#w_h_b = np.random.randn(784,300)
#w_o_b = np.random.randn(300,10)
final_energy = np.zeros([14])
h = [500,150]#[100,300,500,700,1000,1500]
ind=-1
plt.figure()
LR=0.0005


w_h = [np.random.randn(784,h[0]),np.random.randn(h[0],h[1])]
w_o = np.random.randn(h[1],10)
outRes = np.zeros([60000,10,T])
energies = np.zeros([60000])
batch_energy = np.zeros([1200]) # bach_size = 50
ind+=1
#pred_test = np.zeros([1300])
for i in range(60000):
    if (i%1000==0):
        print i
    if (i%5000==0): #5000 
        pred=np.zeros([10000])
        for i2 in range(10000): #10000
            randy  = np.random.random([T])
            spikes2 = np.zeros([784,T])
            for k2 in range(784):
                spikes2[k2,:] = randy<dataTest[i2,k2]
            pred[i2],_,_ = svp.Test(spikes2,T,h,10,w_h,w_o,hiddenThr,outputThr)
        #pred_test[int(i/50)] = sum(pred==labelTest[0:1000])/1000.0
        print(sum(pred==labelTest)/10000.0)
    randy  = np.random.random([T])
    spikes = np.zeros([784,T])
    for k in range(784):
        spikes[k,:] = randy<data[i,k]
    w_h,w_o,outRes[i,:,:],energies[i] = svp.Train(spikes,T,h,10,w_h,w_o,hiddenThr,outputThr,tau,int(label[i]),LR)
   
       
#    pattern=np.zeros([13,2,T*4])
#    index=0
#    select = [0,4,8,12,30,60,100,150,220,300,400,500,600]
#    for i in select:#2000,200):
#        for k in range(4):
#            pattern[index,:,k*T:(k+1)*T]=outRes[i+k,:,:]
#        index+=1
#    plt.figure()
#    lineoffsets1 = [0,1]#np.array(0,1)
#    linelengths1 = .5
#    colors1 = np.array([[1, 0, 0],
#                       [0, 0, 1]])
#    for i in range(13):
#        dat=list()
#        for j in range (2):
#            temp = (np.where(pattern[i,j,:]>0.5)[0]).tolist()
#            dat.append(temp)
#            del temp
#        plt.subplot(13,1,i+1)
#        plt.eventplot(dat, lineoffsets=lineoffsets1,
#                  linelengths=linelengths1,colors=colors1)
#        if i!=12:
#            plt.gca().xaxis.set_major_locator(plt.NullLocator())
#        plt.gca().yaxis.set_major_locator(plt.NullLocator())
#        del dat
    
    
    for i in range(0,60000,50):
        batch_energy[i/50] = np.mean(energies[i:i+50])
    final_energy[ind] = np.mean(batch_energy[1100:1200])
    #plt.plot(batch_energy)
    if (ind%2==0):
        plt.plot(np.convolve(batch_energy,np.ones([3])/3.0,mode='full')[0:1190],label='LR: '+str(LR),linewidth=1.5)
    plt.legend(fontsize=15)
    plt.xlim(1,1500)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xlabel('Iterations',fontsize=20)
    plt.ylabel('MSE',fontsize=20)
    #plt.figure()
    #lr_label = list()
    #for a in range(14):
    #    lr_label.append(str(rates[a]))
    #plt.bar(range(14),final_energy,align='center',color='black')
    #plt.xticks(range(14), lr_label,rotation='vertical')
    #plt.tick_params(axis='both', which='major', labelsize=20)
    #plt.xlabel('Learning Rate',fontsize=20)
    #plt.ylabel('MSE',fontsize=20)
    #plt.subplots_adjust(bottom=0.25)
    
    
    
    pred=np.zeros([10000])
    for i in range(10000):
        randy  = np.random.random([T])
        spikes = np.zeros([784,T])
        for k in range(784):
            spikes[k,:] = randy<dataTest[i,k]
        pred[i],U_o,Spike_o = svp.Test(spikes,T,h,10,w_h,w_o,hiddenThr,outputThr[ind])
    print(sum(pred==labelTest)/10000.0)
    
randy  = np.random.random([T])
spikes = np.zeros([784,T])
for k in range(784):
    spikes[k,:] = randy<data[0,k]
pred,U_o,Spike_o = svp.Test(spikes,T,h,10,w_h,w_o,hiddenThr+50,outputThr[ind]+100)   
U_o = U_o/10.0
U_o[U_o==0]=16
U_o[:,0]=0
U_o[U_o<0] = 0 
U_o[0,7]=0
plt.figure()
plt.plot(U_o[0,0:8].T,'--',Linewidth = 3)
plt.plot(U_o[1:10,0:8].T,Linewidth = 3)
plt.legend(range(0,10),loc='best')
plt.ylim(-1,17)
plt.tick_params(axis='both',labelsize=20)
plt.xlabel('Time (ms)',fontsize=20)
plt.ylabel('Scaled Membrane Potential',fontsize=20)

randy  = np.random.random([T])
spikes = np.zeros([784,T])
for k in range(784):
    spikes[k,:] = randy<data[16,k]
pred,U_o,Spike_o = svp.Test(spikes,T,h,10,w_h,w_o,hiddenThr+50,outputThr[ind]+100)   
U_o = U_o/10.0
for t in range (2,50):
    if U_o[8,t]==0:
        U_o[8,t]=16
U_o[U_o<0] = 0 
U_o[8,10]=0
plt.figure()
plt.plot(U_o[0:8,0:9].T,Linewidth = 3)
plt.plot(U_o[8,0:9].T,'--',Linewidth = 3)
plt.plot(U_o[9:10,0:9].T,Linewidth = 3)
plt.legend(range(0,10),loc='best')
plt.ylim(-1,17)
plt.tick_params(axis='both',labelsize=20)
plt.xlabel('Time (ms)',fontsize=20)
plt.ylabel('Scaled Membrane Potential',fontsize=20)

#fig, ax1 = plt.subplots()
#ax1.plot(pred_test[0:1200]*100, 'b-')
#ax1.set_xlabel('Epoch',fontsize=20)
#
## Make the y-axis label, ticks and tick labels match the line color.
#ax1.set_ylabel('Accuracy %', color='b',fontsize=20)
#ax1.tick_params('y', colors='b', labelsize=20)
#
#ax2 = ax1.twinx()
#ax2.plot(batch_energy, 'r.')
#ax2.set_ylabel('Energy', color='r',fontsize=20)
#ax2.tick_params('y', colors='r', labelsize=20)
#ax1.set_xlim(-20,1200)
#fig.tight_layout()
#plt.show()