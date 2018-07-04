# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:48:29 2017

@author: amir
"""

import numpy as np

def ESPS(spikes,tau):
    n,T = np.shape(spikes)
    newPatch = np.zeros([n,T])
    t_decay = np.zeros([10])
    for t in range(10):
        t_decay[t] = -t
    #t_decay = [0,-1,-2,-3,-4]
    decay = np.exp(t_decay/tau)
    for i in range(n):
        newPatch[i,:] = np.convolve(spikes[i,:],decay,mode='same')
    return newPatch

def LinScale(num):
    if (num>1):
        return 1
    elif (num<-1):
        return -1
    return num

def Train(input_spikes,T,hiddens,outputs,w_h,w_o,threshold_h,threshold_o,tau,label,lr,epsilon=2):
    
    n_hidden = len(hiddens)
    hidden_spikes = [1]*n_hidden
    U_h = [1]*n_hidden
    for i in range(n_hidden):
        hidden_spikes[i]=np.zeros([hiddens[i],T])
    #hidden_spikes = np.zeros([hiddens,T])
    output_spikes = np.zeros([outputs,T])
    for i in range(n_hidden):
        U_h[i] = np.zeros([hiddens[i],T])
    U_o = np.zeros([outputs,T])
    inputs = np.shape(input_spikes)[0]
    energy = 0.0
    
    for t in range(1,T):
        
        U_h[0][:,t] = np.dot(input_spikes[:,t].T,w_h[0]) + U_h[0][:,t-1]#*np.exp(-1/tau)
        for u in range(hiddens[0]):
            if (U_h[0][u,t]>=threshold_h[0]):
                hidden_spikes[0][u,t]=1
                U_h[0][u,t] = 0
        
        for h in range(1,n_hidden):
            U_h[h][:,t] = np.dot(hidden_spikes[h-1][:,t].T,w_h[h]) + U_h[h][:,t-1]#*np.exp(-1/tau)
            for u in range(hiddens[h]):
                if (U_h[h][u,t]>=threshold_h[h]):
                    hidden_spikes[h][u,t]=1
                    U_h[h][u,t] = 0

        U_o[:,t] = np.dot(hidden_spikes[n_hidden-1][:,t].T,w_o) + U_o[:,t-1]#*np.exp(-1/tau)
        
        for o in range(outputs):
            if (U_o[o,t]>=threshold_o):
                output_spikes[o,t] =1
                U_o[o,t] = 0
        delta = np.zeros([outputs,1])
        if (t%epsilon==0):
            if (sum(output_spikes[label,t-epsilon+1:t+1])<1):
                delta[label,0] = 1
                energy+=1
            for l in range(outputs):
                if (l!=label):
                    if (sum(output_spikes[l,t-epsilon+1:t+1])>=1):
                        delta[l,0]=-1
                        energy+=1
            #delta = delta*np.exp(-w_o) # this line is extra
            
            delta_h = [1]*n_hidden
            delta_h[n_hidden-1] = np.dot(w_o,delta)
            w_o += np.dot((sum(hidden_spikes[n_hidden-1][:,t-epsilon+1:t+1].T).T).reshape([hiddens[n_hidden-1],1]),delta.T)*lr
            for der in range(n_hidden-1,0,-1):
                derivative = (sum(hidden_spikes[der][:,t-epsilon+1:t+1].T).T)>0 # or >=0
                delta_h[der] = delta_h[der] * derivative.reshape([hiddens[der],1])
                delta_h[der-1] = np.dot(w_h[der],delta_h[der])                
                w_h[der] += np.dot((sum(hidden_spikes[der-1][:,t-epsilon+1:t+1].T).T).reshape([hiddens[der-1],1]),delta_h[der].T)*lr
                
            derivative = (sum(hidden_spikes[0][:,t-epsilon+1:t+1].T).T)>0 # or >=0
            delta_h[0] = delta_h[0] * derivative.reshape([hiddens[0],1])
            w_h[0] += np.dot((sum(input_spikes[:,t-epsilon+1:t+1].T).T).reshape([inputs,1]),delta_h[0].T)*lr
            
    return w_h,w_o,output_spikes,np.power(energy/(T-1),2)

def Test(input_spikes,T,hiddens,outputs,w_h,w_o,threshold_h,threshold_o):
    n_hidden = len(hiddens)
    hidden_spikes = [1]*n_hidden
    U_h = [1]*n_hidden
    for i in range(n_hidden):
        hidden_spikes[i]=np.zeros([hiddens[i],T])
    #hidden_spikes = np.zeros([hiddens,T])
    output_spikes = np.zeros([outputs,T])
    for i in range(n_hidden):
        U_h[i] = np.zeros([hiddens[i],T])
    U_o = np.zeros([outputs,T])
    
    for t in range(1,T):
        
        U_h[0][:,t] = np.dot(input_spikes[:,t].T,w_h[0]) + U_h[0][:,t-1]#*np.exp(-1/tau)
        for u in range(hiddens[0]):
            if (U_h[0][u,t]>=threshold_h[0]):
                hidden_spikes[0][u,t]=1
                U_h[0][u,t] = 0
        
        for h in range(1,n_hidden):
            U_h[h][:,t] = np.dot(hidden_spikes[h-1][:,t].T,w_h[h]) + U_h[h][:,t-1]#*np.exp(-1/tau)
            for u in range(hiddens[h]):
                if (U_h[h][u,t]>=threshold_h[h]):
                    hidden_spikes[h][u,t]=1
                    U_h[h][u,t] = 0

        U_o[:,t] = np.dot(hidden_spikes[n_hidden-1][:,t].T,w_o) + U_o[:,t-1]#*np.exp(-1/tau)
        
        for o in range(outputs):
            if (U_o[o,t]>=threshold_o):
                output_spikes[o,t] =1
                U_o[o,t] = 0
    return np.argmax(sum(output_spikes.T)),U_o,output_spikes
    
def TrainSoft(input_spikes,T,hiddens,outputs,w_h,w_o,threshold_h,threshold_o,tau,label,lr,epsilon=2):
    hidden_spikes = np.zeros([hiddens,T])
    output_spikes = np.zeros([outputs,T])
    U_h = np.zeros([hiddens,T])
    U_o = np.zeros([outputs,T])
    inputs = np.shape(input_spikes)[0]
    energy = 0.0
    for t in range(1,T):
        U_h[:,t] = np.dot(input_spikes[:,t].T,w_h) + U_h[:,t-1]#*np.exp(-1/tau)
        
        for u in range(hiddens):
            if (U_h[u,t]>=threshold_h):
                hidden_spikes[u,t]=1
                U_h[u,t] = 0
                
        U_o[:,t] = np.dot(hidden_spikes[:,t].T,w_o)# + U_o[:,t-1]#*np.exp(-1/tau)
        U_temp = U_o[:,t]#/hiddens#-np.mean(U_o[:,t])
        
        P_temp = np.exp(U_temp)/sum(np.exp(U_temp))
        #print(P_temp)
        for o in range(outputs):
            if (P_temp[o]>=threshold_o):
                output_spikes[o,t] =1
                #print(P_temp[o],t,o)
                U_o[o,t] = 0
        delta = np.zeros([outputs,1])
        if (t%epsilon==0):
            if (sum(output_spikes[label,t-epsilon+1:t+1])<1):
                delta[label,0] = 1
                energy+=1
            for l in range(outputs):
                if (l!=label):
                    if (sum(output_spikes[l,t-epsilon+1:t+1])>=1):
                        delta[l,0]=-1
                        energy+=1
            delta = delta*np.exp(-w_o)
            derivative = (sum(hidden_spikes[:,t-epsilon+1:t+1].T).T)>0 # or >=0
            delta_h = np.dot(w_o,delta)*derivative.reshape([hiddens,1])
            w_o += np.dot((sum(hidden_spikes[:,t-epsilon+1:t+1].T).T).reshape([hiddens,1]),delta.T)*lr#*np.exp(-w_o)
            w_h += np.dot((sum(input_spikes[:,t-epsilon+1:t+1].T).T).reshape([inputs,1]),delta_h.T)*lr
    print(sum(output_spikes.T),label)
               
    return w_h,w_o,output_spikes,np.power(energy/(T-1),2)

def TestSoft(input_spikes,T,hiddens,outputs,w_h,w_o,threshold_h,threshold_o):
    hidden_spikes = np.zeros([hiddens,T])
    output_spikes = np.zeros([outputs,T])
    U_h = np.zeros([hiddens,T])
    U_o = np.zeros([outputs,T])
    inputs = np.shape(input_spikes)[0]
    energy = 0.0
    for t in range(1,T):
        
        U_h[:,t] = np.dot(input_spikes[:,t].T,w_h) + U_h[:,t-1]#*np.exp(-1/tau)
        
        for u in range(hiddens):
            if (U_h[u,t]>=threshold_h):
                hidden_spikes[u,t]=1
                U_h[u,t] = 0
                #esps_hidden[u,t]+= 1
        U_o[:,t] = np.dot(hidden_spikes[:,t].T,w_o) #+ U_o[:,t-1]#*np.exp(-1/tau)
        
        U_temp = U_o[:,t]#/hiddens#-np.mean(U_o[:,t])
        P_temp = np.exp(U_temp)/sum(np.exp(U_temp))
        for o in range(outputs):
            if (P_temp[o]>=threshold_o):
                output_spikes[o,t] =1
                U_o[o,t] = 0
    #print(sum(output_spikes.T))
    return np.argmax(sum(output_spikes.T))
def Train2(inputs,w_h,w_o,label):
    hidden = np.dot(w_h.T,inputs.reshape([2,1]))
    #print(hidden)
    derivative = hidden>=0 #.2
    hidden[hidden<0]=0 #.2
    #print(hidden)
    output = np.maximum(0,np.dot(w_o.T,hidden))
    target = np.zeros([2,1])
    target[label]=1
    delta = (target-output)
    #print(np.shape(w_o),np.shape(delta),np.shape(derivative))
    delta_h = np.dot(w_o,delta)*derivative
    w_o = w_o + 0.0005*np.dot(hidden,delta.T)
    w_h = w_h + 0.0005*np.dot(inputs.reshape([1,2]).T,delta_h.T)
    return w_h,w_o
    
    