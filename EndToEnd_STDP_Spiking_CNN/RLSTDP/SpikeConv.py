import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from six.moves import cPickle as pickle
import multiprocessing as mp
import random as rnd
import math

def Read_Data(data_name,lab=True):
    if (lab):
        with open(data_name,'rb') as f:
                save = pickle.load(f)
                data=save['Data']
                label=save['Label']
        return data,label
    else:
        with open(data_name,'rb') as f:
                save = pickle.load(f)
                data=save['image']
        return data

def Image_Read(im_name,ncol=0):
        #print(im_name.shape)
        if (type(im_name)==str):
                im = mpimg.imread(im_name)
        else:
                im = np.reshape(im_name, (-1, ncol))                       
	return im
 
def Show_Weight(oneData):
        dimy=np.shape(oneData)[0]
        dim1=int(np.sqrt(dimy))
        tempPic=np.zeros([dim1,dim1])
        for i in range(0,dim1):
                for j in range(0,dim1):
                        tempPic[i,j]=oneData[i*dim1+j]
        return tempPic
	
def Image_Patch(im,p):
    
	n = np.shape(im)[0]
	m = np.shape(im)[1]
	
	r = n-p+1
	c = m-p+1
	
	im_patches=list()
	
	for i in range(r):
		for j in range(c):
			im_patches.append(im[i:i+p,j:j+p])
	
	return im_patches
 
def FeatureMap_Patch(spikes,p, overlap=True,hidden=False):
    n = np.shape(spikes)[0]
    m = np.shape(spikes)[1]
    T = np.shape(spikes)[2]
    im_patches=list()
    
    if (overlap):
#        if (hidden):
#            zPad_spikes = spikes
#        else:
        zPad_spikes = np.zeros([n+2*(int)(p/2),m+2*(int)(p/2),T])
        zPad_spikes[(int)(p/2):n+(int)(p/2),(int)(p/2):m+(int)(p/2),:]=spikes
        
        
        for i in range(n):
            for j in range(m):
                im_patches.append(zPad_spikes[i:i+p,j:j+p,:])
    else:
        new_n=n
        new_m=m
        if (n%p!=0):
            new_n = n+p-(n%p)
        if (m%p!=0):
            new_m = m+p-(m%p)
        zPad_spikes=np.zeros([new_n,new_m,T])
        zPad_spikes[0:n,0:m,:]=spikes
        for i in range(new_n/p):
            for j in range(new_m/p):
                im_patches.append(zPad_spikes[i*p:(i+1)*p,j*p:(j+1)*p,:])
    return im_patches,zPad_spikes

def Normalize(Data):
        points = np.shape(Data)[0]
        classes = np.shape(Data)[1]
        new_data = np.zeros([points,classes])
        for i in range(classes):
                new_data[:,i]=(Data[:,i]-np.mean(Data[:,i]))
        return new_data

	
def Update_Zylberg(w,q,thr,unitspike,x):
    r1=0.01
    r2=0.0001
    r3=0.02    
    p=0.05
    d=np.shape(x)[0]
    neurons = np.shape(unitspike)[0]
    n=np.zeros([neurons])
    
    for u in range(neurons):
        n[u] = sum(unitspike[u,:])
    #print(n)
    for u1 in range(neurons):
        for u2 in range(neurons):
            if (u1!=u2):
                w[u1,u2] += r1*(n[u1]*n[u2]-np.power(p,2))
    
    for k in range(d):
        q[k,:] += r2*n*(x[k]-n*q[k,:])
        
    thr += r3*(n-p)
    
    return w,q,thr

def Train_Convolution_Patch_Zylberg(X, Q, W, THR, T, units):
	D = np.shape(X)[0]
	
	c=1
	r=1
	tau=r*c

	unitSpikes = np.zeros([units,T])
	I = np.zeros([T,units])
	U = np.zeros([T,units])
	tempU=np.dot(X,Q) #1*d . d*units = 1*units
	#print(tempU)
	for t in range(1,T):
		tempInh = np.zeros([units])
		for u in range(0,units):
			tempInh += W[:,u]*unitSpikes[u,t-1]  # 1*units
		I[t,:]=tempU-tempInh
		#print(I)
		v_inf=I[t,:]*r
		U[t,:]=v_inf+(U[t-1,:]-v_inf)*np.exp(-1/tau)
		#print(U)
		for u in range(0,units):
			if (U[t,u]>=THR[u]):
				U[t,u]=0
				#print(i,"YES")
				unitSpikes[u,t]=1
	W,Q,THR = Update_Zylberg(W,Q,THR,unitSpikes,X)
	return Q,W,THR

def STDPProb(z,w,ins):
    mu=0.001 # smaller rate than single spike .001
    w2=w
    d=np.shape(ins)[0]
    for i in range(0,d):
        if (ins[i]==1):
            w2[i,z] = min(w[i,z]+mu*(np.exp(-w[i,z])-np.exp(-1)),1)
            #w2[i,z] = min(w[i,z]+mu*(np.exp(-w[i,z])-np.exp(-1)),1)
        else:
            w2[i,z]=max(w[i,z]-mu*.75*(np.exp(w[i,z])-1),0) # or ,mu
            #w2[i,z]=max(w[i,z]-mu*.75*(np.exp(w[i,z])-1),0) # or ,mu
    return w2
    
def STDPLagrange(z,w,ins,landa=0):
    mu=0.0005# the current is only for 32 Nature to see    0.0002#0.001#0.01 (This is for Epsilon)#0.0005 # smaller rate than single spike .001
    w2=w
    scale=0.75
    d=np.shape(ins)[0]
    for i in range(0,d):
        if (ins[i]==1):
            w2[i,z] = min(w[i,z]+mu*(1-w2[i,z]*(1+landa)),1)
            #w2[i,z] = min(w[i,z]+mu*(np.exp(-w[i,z])-np.exp(-1)),1)
        else:
            w2[i,z]=max(w[i,z]-mu*scale*(w2[i,z]*(1+landa)),0) # or ,mu
            #w2[i,z]=max(w[i,z]-mu*.75*(np.exp(w[i,z])-1),0) # or ,mu
    return w2

def STDPProbG(z,w,ins):
    mu=0.001 # smaller rate than single spike .001
    w2=w
    d=np.shape(ins)[0]
    for i in range(0,d):
        if (ins[i]==1):
            w2[i,z] = w[i,z]+mu*(np.exp(-w[i,z]))
        else:
            w2[i,z]=w[i,z]-mu*.75 # or ,mu
    return w2

def Poisson_2D(data,T,maxrate = 1):
        dat = data/(255.0/maxrate)
        s = np.shape(data)[0]
        s2 = np.shape(data)[1]
        spike_train = np.zeros([s,s2,T])
        for i in range(s):
            for j in range(s2):
                randy = np.random.random([T])
                for k in range(T):
                        if dat[i,j]>=randy[k]:
                                spike_train[i,j,k]=1
        return spike_train

def ISI_2D(data,T,maxrate = 10):
        delay = 2
        s = np.shape(data)[0]
        s2 = np.shape(data)[1]
        spike_train = np.zeros([s,s2,T])
        for i in range(s):
            for j in range(s2):
                spike_count = int(round(data[i,j]/(255.0)*maxrate))
                if (spike_count>0):
                    gap = T/spike_count
                    for k in range(spike_count):
                        spike_train[i,j,k*gap+delay]=1
        return spike_train

def Autoencoder_L1_Train(spikes,units,patch_size,w):
    threshold = 0.15 #0.25
    spike_patches,_ = FeatureMap_Patch(spikes,patch_size)
    T = np.shape(spikes)[2]
    for patchy in spike_patches:
        if (np.sum(patchy)==0):
            continue
        U=np.zeros([units,T])
        
        for t in range(1,T):
            y_temp = patchy[:,:,t]
            y_t = y_temp.flatten()
            U[:,t] = np.dot(w.T,y_t)#+U[:,t-1]
            U_temp = U[:,t]-np.mean(U[:,t])
            P = np.exp(U_temp)/(np.sum(np.exp(U_temp)))
            
            
            for j in range(units):
                if (P[j]>=threshold):
                    #print(P[j],t,j)
                    w = STDPProb(j,w,y_t)
                    break;
    return w
    
def Autoencoder_L1_Train_Thr(spikes,units,patch_size,w,threshold_glob,landa=0,tau=0.001,epsilon=0,inh = 0):
    threshold_rate = 0.00005#0.0001 #0.001 for Epsilon
    #threshold_glob = 0.1 #0.25
    spike_patches,_ = FeatureMap_Patch(spikes,patch_size)
    T = np.shape(spikes)[2]
    for patchy in spike_patches:
        if (np.sum(patchy)==0):
            continue
        U=np.zeros([units,T])
        patchy2 = ESPS(patchy,tau)
        inhibition = 0
        #m = np.zeros([units]) for epsiln
        for t in range(1,T):
            slot = max(0,t-epsilon)
            y_slot = np.max(patchy[:,:,slot:t+1],axis=2)
            y_spikes = y_slot.flatten()
            y_temp = patchy2[:,:,t]
            y_t = y_temp.flatten()
            #y_spikes = patchy[:,:,t].flatten()
            U[:,t] = np.dot(w.T,y_t)#+U[:,t-1]
            U_temp = U[:,t]-np.mean(U[:,t])
            P = np.exp(U_temp)/(np.sum(np.exp(U_temp)))
            P = P-inhibition
            m = np.zeros([units]) # for dense 
            for j in range(units):
                if (P[j]>=threshold_glob):
                    #print(P[j],t,j)
                    w = STDPLagrange(j,w,y_spikes,landa)
                    m[j]=1
                    inhibition = inh
                else:
                    inhibition = inhibition*np.exp(-1)
            threshold_glob+= (threshold_rate*(sum(m)-1))  # ahead when no epsilon
                    
    return w,threshold_glob

def ESPS(aPatch,tau):
    n,m,T = np.shape(aPatch)
    newPatch = np.zeros([n,m,T])
    t_decay = np.zeros([10])
    for t in range(10):
        t_decay[t] = -t
    #t_decay = [0,-1,-2,-3,-4]
    decay = np.exp(t_decay/tau)
    for i in range(n):
        for j in range(m):
            newPatch[i,j,:] = np.convolve(aPatch[i,j,:],decay,mode='same')
    return newPatch

def Autoencoder_Hidden_Train(fmaps,w,p,units,thr,randy=3.0,tau=0.001,epsilon=0,inh = 0):
    threshold_rate = 0.00001#0.00001
    in_patches = list()
    D1 = len(fmaps)
    T=np.shape(fmaps[0])[2]
    
    for d in range(D1):
        patches,new_spikes = FeatureMap_Patch(fmaps[d],p,True,True)
        in_patches.append(patches)
        del patches
    L=len(in_patches[0])
    print(L)
    for l in range(L):
        if ((l+1)%randy!=0.0):
            #print(l)
            continue
        for t in range(1,T):
            U = np.zeros([units,T])
            y_t = np.zeros([p*p*D1])
            for d1 in range(D1):
                y_t[d1*p*p:p*p*(d1+1)] = in_patches[d1][l][:,:,t].flatten()
            U[:,t] = np.dot(w.T,y_t)#+U[:,t-1]
            if (np.sum(U[:,t])==0):
                continue
            U_temp = U[:,t]-np.mean(U[:,t])
            P = np.exp(U_temp)/(np.sum(np.exp(U_temp)))
            
            m = np.zeros([units])
            for j in range(units):
                if (P[j]>=thr):
                    #print(P[j],t,j)
                    w = STDPLagrange(j,w,y_t)
                    m[j]=1
            thr+= (threshold_rate*(sum(m)-1))
                    
    return w,thr
            
def Convolution_L1(spikes,w,p,T,threshold,n,m):
        #depth = 1#np.shape(w)[1]

        patches,_ = FeatureMap_Patch(spikes,p)
        patch_no = len(patches)

        u_tot = np.zeros([patch_no])#depth,...
        post_spikes = np.zeros([patch_no,T])#depth,...
        
        
        for j in range(patch_no):
                U = np.zeros([T])
                for t in range(1,T):
                        patchy = patches[j][:,:,t].flatten()
                        if (np.sum(patchy)==0):
                            U[t] = (U[t-1]*np.exp(-1))
                        else:
                            U[t] = np.dot(w.T,patchy)/(np.linalg.norm(w)*np.linalg.norm(patchy)) + (U[t-1]*np.exp(-1))
                        if (U[t]>=threshold):
                                print(U[t],t,j)
                                U[t]=0
                                post_spikes[j,t] = 1
                        if (np.sum(patchy)!=0):
                            u_tot[j]+=np.dot(w.T,patchy)/(np.linalg.norm(w)*np.linalg.norm(patchy))#np.sum(w)#(np.linalg.norm(w))
        return post_spikes,two_D(u_tot,n,m)    
    
def STDPSigmoid(z,w,ins):
    mu=0.001
    w2=w
    d=np.shape(ins)[0]
    for i in range(0,d):
        if (ins[i]==1):
            w2[i,z] = min(w[i,z]+mu*(w[i,z]*(1-w[i,z])),1)
        else:
            w2[i,z]=max(w[i,z]-mu*.75*(w[i,z]*(1-w[i,z])),0) # or ,mu
    return w2
    
def STDPBasic(z,w,ins,t_f):
    mu=0.01
    w2=w
    d=np.shape(ins)[0]
    for i in range(0,d):
        for t in range(0,5):
            if (ins[i,t]==1):
                w2[i,z] = min(w[i,z]+mu*(np.exp(-(t_f-t)/2)),1)
            else:
                w2[i,z]=max(w[i,z]-mu*.75*(np.exp(-2)),0) # or ,mu
    return w2

def Train_Classifier(w,spikes):
    mu=0.000005
    D=np.shape(spikes)[0]
    for d in range(D):
        if (spikes[d]==1):
            w[d]+=(mu*(np.exp(-w[d])-np.exp(-1)))
        else:
            w[d]-=(mu*.75*(np.exp(w[d])-1))
    return w
    
def Classifier(pre_spikes,unit,w_unit,threshold):
    # w_unit is hidden*1,  pre_spike: hidden*T,   unit is a number
    T=np.shape(pre_spikes)[1]
    
    for t in range(T):
        U = np.dot(w_unit,pre_spikes[:,t])
        if (U>threshold):
            w_unit = Train_Classifier(w_unit,pre_spikes[:,t])
    return w_unit
    
def ClassifierTest(pre_spikes,w):
    # w_unit is hidden*1,  pre_spike: hidden*T,   unit is a number
    T=np.shape(pre_spikes)[1]
    U = np.zeros([10])
    for t in range(T):
        U+= np.dot(w.T,pre_spikes[:,t])
    return np.argmax(U)

def Hidden_NN(spike_trains,exW,thr,Outs):
    # for each sample
    D=np.shape(spike_trains)[0]
    T=np.shape(spike_trains)[1]
    I=np.zeros([Outs,T])
    U_tot = np.zeros([Outs])
    post_spike = np.zeros([Outs,T])
    for t in range(1,T):
        I[:,t] = I[:,t-1]*np.exp(-1)
        preSpike = spike_trains[:,t].T
        U = np.dot(exW.T,preSpike)-I[:,t]
        #print(U)
        if (sum(U)==0):
            continue
        U_tot+=U
        U=U-D*.05
        #print(U)
        P=np.exp(U)/np.sum(np.exp(U))
        #print(P)
        accp=np.cumsum(P)
        randZ=rnd.random()
        for unit in range(0,Outs):
            if (accp[unit]>=randZ and P[unit]>=thr):
                post_spike[unit,t] = 1
                temp_I = I[unit,t]
                I[:,t]+=.5
                I[unit,t] = temp_I
                break
        
        del preSpike
        del accp
    return U_tot,post_spike
    
def Hidden_NN2(spike_trains,exW,thr,Outs):
    # for each sample
    D=np.shape(spike_trains)[0]
    T=np.shape(spike_trains)[1]
    I=np.zeros([Outs,T])
    U_tot = np.zeros([Outs])
    post_spike = np.zeros([Outs,T])
    for t in range(1,T):
        I[:,t] = I[:,t-1]*np.exp(-1)
        preSpike = spike_trains[:,t].T
        U = np.dot(exW.T,preSpike)-I[:,t]
        #print(U)
        if (sum(U)==0):
            continue
        U_tot+=U
        #U=U-D*.05
        #print(U)
        P=U/np.sum(U)
        print(P)
        #accp=np.cumsum(P)
        #randZ=rnd.random()
        for unit in range(0,Outs):
            if (P[unit]>=thr):
                post_spike[unit,t] = 1
                temp_I = I[unit,t]
                I[:,t]+=.5
                I[unit,t] = temp_I
        
        del preSpike
        #del accp
    return U_tot,post_spike

def Hidden_NN4(spike_trains,exW,thr,Outs):
    # for each sample
    #D=np.shape(spike_trains)[0]
    T=np.shape(spike_trains)[1]
    sum_spike = sum(spike_trains.T)
    U_tot = np.zeros([Outs])
    
    post_spike = np.zeros([Outs,T])
    
    for i in range(Outs):
        U_tot[i]=(np.dot(exW[:,i],sum_spike)/(np.sqrt(sum(np.power(exW[:,i],2)))))  # division is added
    
    return U_tot,post_spike
   
def Hidden_NN3(spike_trains,exW,thr,Outs):
    # for each sample
    D=np.shape(spike_trains)[0]
    T=np.shape(spike_trains)[1]
    U_tot = np.zeros([Outs])
    post_spike = np.zeros([Outs,T])
    U = np.zeros([Outs,T])
    
    for t in range(1,T):
        preSpike = spike_trains[:,t].T
        temp = np.dot(exW.T,preSpike)/D
        U[:,t] = temp+U[:,t-1]*np.exp(-1)
        U_tot+=(np.dot(exW.T,preSpike)/(sum(exW)))  # division is added
        for unit in range(0,Outs):
            if (U[unit,t]>=thr):
                post_spike[unit,t] = 1
                U[unit,t] = 0
        
        del preSpike
        #del accp
    return U_tot,post_spike

def NN_Layer(spike_trains,T,exW,thr,Outs,func):
    R=np.shape(spike_trains)[0]
    D=np.shape(spike_trains)[1]
    for row in range(0,R):
        if (row%100==0):
            print(" ********* "+str(row)+" images are done **********")
        I=np.zeros([Outs,T])
        for t in range(1,T):
            I[:,t] = I[:,t-1]*np.exp(-1)
            preSpike = spike_trains[row,:,t].T
            U = np.dot(exW.T,preSpike)-I[:,t]
            
            if (sum(U)==0):
                continue
            U=U-D*.05
            #print(U)
            P=np.exp(U)/np.sum(np.exp(U))
            #print(P)
            accp=np.cumsum(P)
            randZ=rnd.random()
            UnitIndex = -1
            for unit in range(0,Outs):
                if (accp[unit]>=randZ and P[unit]>=thr):
                    #print("It is here")
                    UnitIndex=unit
                    #print(unit)
                    temp_I = I[unit,t]
                    I[:,t]+=.5
                    I[unit,t] = temp_I
                    break
            if (UnitIndex!=-1):
                exW = func(UnitIndex,exW,preSpike)
            del preSpike
            del accp
    return exW

def NN_Layer_2(spike_trains,T,exW,thr,Outs,func):
    R=np.shape(spike_trains)[0]
    D=np.shape(spike_trains)[1]
    for row in range(0,R):
        if (row%100==0):
            print(" ********* "+str(row)+" images are done **********")
        
        for t in range(1,T):
            
            preSpike = spike_trains[row,:,t].T
            U_1 = np.dot(exW.T,preSpike)
            
            if (np.sum(U_1)==0):
                continue
            
            U = (U_1-np.mean(U_1))     
            
            #U=U-D*.04
            #print(U)
            P=np.exp(U)/np.sum(np.exp(U))
            #print(P)
            randZ = np.random.random([Outs])
            for unit in range(0,Outs):
                if (P[unit]>=randZ[unit] and P[unit]>=thr):
                    #print("It is here")
                    #print(unit,P[unit],row)
                    exW = func(unit,exW,preSpike)
                    
            del preSpike
    return exW

def NN_Layer_RBM(spike_trains,T,exW,thr,Outs,func):
    R=np.shape(spike_trains)[0]
    D=np.shape(spike_trains)[1]
    
    for row in range(0,R):
        U=np.zeros([Outs,T])
        if (row%100==0):
            print(" ********* "+str(row)+" images are done **********")
            #thr= thr+1
        spike_normilizer = np.sum(spike_trains[row,:,:])/D
        for t in range(1,T):
            preSpike = spike_trains[row,:,t].T
            U[:,t] = np.dot(exW.T,preSpike)+U[:,t-1]*np.exp(-0.5)
            temp = U[:,t]/spike_normilizer
            #print(row)
            #print(temp)
            for unit in range(0,Outs):
                if (temp[unit]>=thr):
                    U[unit,t]=0
                    #print(unit,temp[unit],t,row,thr)
                    exW = func(unit,exW,preSpike)
            del preSpike
            del temp
    return exW,thr

def Poisson(data,T,dev=255.0):
        dat = data/dev
        s = np.shape(data)[0]
        spike_train = np.zeros([s,T])
        for i in range(s):
                randy = np.random.random([T])
                for j in range(T):
                        if dat[i]>=randy[j]:
                                spike_train[i,j]=1
        return spike_train

def Conv_Sec(image,w,p,T,threshold,n2,m2,stride,new_n2,new_m2):
        #depth = 1#np.shape(w)[1]

        patches = Image_Patch(image,p)
        patch_no = len(patches)

        u_tot = np.zeros([patch_no])#depth,...
        spikes = np.zeros([patch_no,T])#depth,...
        
        
        for j in range(patch_no):
                U = np.zeros([T])
                patchy = patches[j].flatten()
                pre_spikes = Poisson(patchy,T)
                for t in range(1,T):
                        temp_spike = pre_spikes[:,t]
                        U[t] = np.dot(w,temp_spike) + (U[t-1]*np.exp(-1))
                        if (U[t]>=threshold):
                                U[t]=0
                                #self._spikes[i,j,t] = 1
                                #print("come on")
                                spikes[j,t] = 1
                        #self._u_tot[i,j]+=np.dot(self._w[:,i],temp_spike)
                        u_tot[j]+=np.dot(w,temp_spike)
        
        feature_maps,pooled_feature_maps = Pooling(spikes,n2,m2,stride,new_n2,new_m2)
        return spikes,feature_maps,pooled_feature_maps

def two_D(vector,n2,m2):
        new_map = np.zeros([n2,m2])
        for i in range(n2):
                for j in range(m2):
                        #print(vector[i*m2+j])
                        new_map[i,j] = vector[i*m2+j]
        return new_map

def two_D_spike(vector,n2,m2):
        TT = np.shape(vector)[1]
        new_map = np.zeros([n2,m2,TT])
        for i in range(n2):
                for j in range(m2):
                        #print(vector[i*m2+j])
                        new_map[i,j,:] = vector[i*m2+j,:]
        return new_map

def MaxPool(fmaps,stride):
    L = len(fmaps)
    n,m,T=np.shape(fmaps[0])
    n_new = n+(n%stride)
    m_new = m+(m%stride)
    feature_map=list()
    pool_map=list()
    maps=list()
    for l in range(L):
        f_m,p_m = Pooling(fmaps[l],n,m,stride,n_new,m_new,True)
        feature_map.append(f_m)
        pool_map.append(p_m)
        maps.append((sum(p_m.T).T).squeeze())
        del f_m
        del p_m
    return feature_map,pool_map,maps # I changed this to sum(...) Dec 2017 for DCSNN Feature Extraction
    
def Pooling(u_tot,n2,m2,stride,new_n2,new_m2,DD=False):
        #depth = 1#np.shape(u_tot)[0]
        #feature_maps=list()
        #pooled_feature_maps = list()
        
        if (DD==False):
            feuture_map_flatten = u_tot
            map_temp = two_D_spike(feuture_map_flatten,n2,m2)
            feature_maps = two_D(sum(feuture_map_flatten.T),n2,m2)
            TT = np.shape(map_temp)[2]
        else:
            map_temp = u_tot
            feature_maps = sum(u_tot.T).T
            TT = np.shape(u_tot)[2]

        pooled_map_temp = np.zeros([int(new_n2/stride),int(new_m2/stride),TT])

        temp_2d_map = np.zeros([new_n2,new_m2,TT])
        temp_2d_map[0:n2,0:m2,:]=map_temp
        #print(temp_2d_map[0:10,0:10])
        index_n = 0
        
        while (index_n<=new_n2-stride):
                index_m = 0
                while (index_m<=new_m2-stride):
                        max_temp=np.zeros([TT])
                        for s1 in range(stride):
                            for s2 in range(stride):
                                if (sum(max_temp)<=sum(temp_2d_map[index_n+s1,index_m+s2,:])):
                                    max_temp = temp_2d_map[index_n+s1,index_m+s2,:]
                        pooled_map_temp[int(index_n/stride),int(index_m/stride),:]=max_temp
                        index_m+=stride
                index_n+=stride
        
        pooled_feature_maps = pooled_map_temp
        return feature_maps,pooled_feature_maps

def ImageConstruct(recon,p,n,m):
    r=n/p
    c=m/p
    
    if (len(np.shape(recon))>2):
        recon_image=np.zeros([n,m,np.shape(recon)[2]])
        index=0
        for r1 in range(r):
            for c1 in range(c):
                recon_image[r1*p:(r1+1)*p,c1*p:(c1+1)*p,:]=two_D_spike(recon[index,:,:],p,p)
                index+=1
    else:
        recon_image=np.zeros([n,m])
        index=0
        for r1 in range(r):
            for c1 in range(c):
                recon_image[r1*p:(r1+1)*p,c1*p:(c1+1)*p]=two_D(recon[index,:],p,p)
                index+=1
    return recon_image

def Kurtosis (X):
    mu=np.mean(X)
    std=np.std(X)
#    prev=0.0
#    for xi in X:
#        prev+=math.pow(xi-mu,4)
#    prev=prev/len(X)
#    try:
#        prev=prev/math.pow(std,4)
#    except:
#        return len(X)-3
#    return prev-3
    c_k = std/mu
    return 1.0/((c_k**2)+1)
 
def Sparsity(filters, image_spike, p, threshold, T, overlap=False):
    Patches,new_spike = FeatureMap_Patch(image_spike,p,overlap)
    units = np.shape(filters)[1]
    post_spikes = np.zeros([len(Patches),units,T])
    index=-1
    zero_patch = 0
    k_sparsity=list()
    for patchy in Patches:
        index+=1
        if (np.sum(patchy)==0):
            zero_patch+=1
            continue
        
        for u in range(units):
            U=np.zeros([T])
            for t in range (1,T):
                patch_vec = patchy[:,:,t].flatten()
                if (np.sum(patch_vec)==0):
                    U[t] = U[t-1]*np.exp(-1)
                else:
                    U[t] = np.dot(filters[:,u].T,patch_vec)/(np.linalg.norm(filters[:,u])*np.linalg.norm(patch_vec))+(U[t-1]*np.exp(-1))
                if (U[t]>=threshold):
                    #print(U[t],t,j)
                    U[t]=0
                    post_spikes[index,u,t] = 1
        if (np.sum(post_spikes[index,:,:])!=0):
            #print(index,np.sum(post_spikes[index,:,:]))
            k_sparsity.append(Kurtosis((sum(post_spikes[index,:,:].T))))
        
    sparsity_val = np.sum(post_spikes)/(index-zero_patch+1)/(units*T)
    k_sparsity_val = np.sum(k_sparsity)/(index-zero_patch+1)
    initial_spike = np.sum(image_spike)/(index-zero_patch+1)/(T*p*p)
    return post_spikes,sparsity_val,initial_spike,k_sparsity_val
       
def Reconstruct(filters, image_spike, p, threshold, T, overlap=False,isConvolve=False,tau=1,inhibition = 0):
    Patches,new_spike = FeatureMap_Patch(image_spike,p,overlap)
    n=np.shape(new_spike)[0]
    m=np.shape(new_spike)[1]
    units = np.shape(filters)[1]
    representations = np.zeros([len(Patches),units])
    post_spikes = np.zeros([len(Patches),units,T])
    index=-1
    for patchy in Patches:
        index+=1
#        if (index%100==0):
#            print(index)
        for u in range(units):
            U=np.zeros([T])
            inh = 0
            for t in range (1,T):
                patch_vec = patchy[:,:,t].flatten()
                if (np.sum(patch_vec)==0):
                    U[t] = U[t-1]*np.exp(-1.0/tau)
                else:
                    #this is for Representation Learning (You can add 0.0000001 to the denominator)
                    U[t] = np.dot(filters[:,u].T,patch_vec)/(np.linalg.norm(filters[:,u])*np.linalg.norm(patch_vec)+0.000000000000000001)+(U[t-1]*np.exp(-1.0/tau))
                    #This is for test of DCNN                    
                    #U[t] = np.dot(filters[:,u].T,patch_vec)/(np.linalg.norm(filters[:,u]))+(U[t-1]*np.exp(-1.0/tau))
                    U[t] = U[t]-inh # just added
                if (U[t]>=threshold):
                    #print(U[t],t,u)
                    inh = inhibition
                    U[t]=0
                    post_spikes[index,u,t] = 1
                else:
                    inh = inh*np.exp(-1)
                if (np.sum(patch_vec)!=0):
                    representations[index,u]+=np.dot(filters[:,u].T,patch_vec)
    #print(np.shape(post_spikes))
    if (isConvolve):
        repre_spike = list()
        for u_map in range(units):
            #print(np.shape(post_spikes[:,u_map,:]))
            repre_spike.append(two_D_spike(post_spikes[:,u_map,:],np.shape(image_spike)[0],np.shape(image_spike)[1]))
        
        return repre_spike #,post_spikes # a list of 2D maps  I added post_spikes on August 27th  ,
    
    index+=1
    recon_spikes = np.zeros([index,p*p,T])
    reconstructions = np.zeros([index,p*p])
    for i in range(index):
#        if (index>100 and i%100==0):
#            print(i)
        U=np.zeros([T])
        for point in range(p*p):
            inh = 0
            for t in range(1,T):
                map_vec = post_spikes[i,:,t]
                U[t] = np.dot(filters[point,:],map_vec)+(U[t-1]*np.exp(-1.0/tau))
                if (U[t]>=threshold):
                    inh = inhibition
                    #print(U[t],t,j)
                    U[t]=0
                    recon_spikes[i,point,t] = 1
                else:
                    inh = inh*np.exp(-1)
                reconstructions[i,point]+=np.dot(filters[point,:],map_vec)
    
    
    if (overlap):
        overlap_recon_image = np.zeros([n,m])
        overlap_recon_spike = np.zeros([n,m,T])
        
        for i1 in range(0,n-2*int(p/2)):
            for j1 in range(0,m-2*int(p/2)):
                #print(np.shape(two_D(reconstructions[i1*p+j1,:],p,p)))
                overlap_recon_image[i1:i1+p,j1:j1+p]+=two_D(reconstructions[i1*(n-2*int(p/2))+j1,:],p,p)/(p*p)
                overlap_recon_spike[i1:i1+p,j1:j1+p,:]+=two_D_spike(recon_spikes[i1*(n-2*int(p/2))+j1,:,:],p,p)/(p*p)
    else:
        overlap_recon_image = np.zeros([n,m])
        overlap_recon_spike = np.zeros([n,m,T])
        overlap_recon_image = ImageConstruct(reconstructions,p,n,m)
        overlap_recon_spike = ImageConstruct(recon_spikes,p,n,m)
    
    return new_spike,representations,overlap_recon_image,overlap_recon_spike,Patches,recon_spikes
    
def NewReconstruct(filters, image_spike, p, threshold, T, overlap=False,isConvolve=False,tau=1,inhibition = 0):
    Patches,new_spike = FeatureMap_Patch(image_spike,p,overlap)
    n=np.shape(new_spike)[0]
    m=np.shape(new_spike)[1]
    units = np.shape(filters)[1]
    representations = np.zeros([len(Patches),units])
    post_spikes = np.zeros([len(Patches),units,T])
    index=-1
    for patchy in Patches:
        index+=1
        patchy2 = ESPS(patchy,tau)
#        if (index%100==0):
#            print(index)
        inh = 0
        for t in range (1,T):
            patch_vec = patchy2[:,:,t].flatten()
            if (np.sum(patch_vec)==0):
                continue 
            U = np.dot(filters.T,patch_vec)
            U_temp = U-np.mean(U)
            P = np.exp(U_temp)/(np.sum(np.exp(U_temp)))
            P = P-inh
            for u in range(units):
                if (U[u]>=threshold):
                    #print(U[t],t,u)
                    inh = inhibition
                    #U[u]=0
                    post_spikes[index,u,t] = 1
                else:
                    inh = inh*np.exp(-1)
                if (np.sum(patch_vec)!=0):
                    representations[index,u]+=np.dot(filters[:,u].T,patch_vec)
    #print(np.shape(post_spikes))
    if (isConvolve):
        repre_spike = list()
        for u_map in range(units):
            #print(np.shape(post_spikes[:,u_map,:]))
            repre_spike.append(two_D_spike(post_spikes[:,u_map,:],np.shape(image_spike)[0],np.shape(image_spike)[1]))
        
        return repre_spike # a list of 2D maps
    
    index+=1
    recon_spikes = np.zeros([index,p*p,T])
    reconstructions = np.zeros([index,p*p])
    for i in range(index):
        #if (index>100 and i%100==0):
            #print(i)
        #U=np.zeros([T])
        #for point in range(p*p):
        inh = 0
        post_2 = ESPS(post_spikes,tau)
        
        for t in range(1,T):
            map_vec = post_2[i,:,t]
            U = np.dot(filters,map_vec)
            U_temp = U-np.mean(U)
            P = np.exp(U_temp)/(np.sum(np.exp(U_temp)))
            P = P-inh
            for point in range(p*p):
                if (U[point]>=threshold):
                    inh = inhibition
                    #print(U[t],t,j)
                    #U[t]=0
                    recon_spikes[i,point,t] = 1
                else:
                    inh = inh*np.exp(-1)
                reconstructions[i,point]+=np.dot(filters[point,:],map_vec)
    
    
    if (overlap):
        overlap_recon_image = np.zeros([n,m])
        overlap_recon_spike = np.zeros([n,m,T])
        
        for i1 in range(0,n-2*int(p/2)):
            for j1 in range(0,m-2*int(p/2)):
                #print(np.shape(two_D(reconstructions[i1*p+j1,:],p,p)))
                overlap_recon_image[i1:i1+p,j1:j1+p]+=two_D(reconstructions[i1*(n-2*int(p/2))+j1,:],p,p)/(p*p)
                overlap_recon_spike[i1:i1+p,j1:j1+p,:]+=two_D_spike(recon_spikes[i1*(n-2*int(p/2))+j1,:,:],p,p)/(p*p)
    else:
        overlap_recon_image = np.zeros([n,m])
        overlap_recon_spike = np.zeros([n,m,T])
        overlap_recon_image = ImageConstruct(reconstructions,p,n,m)
        overlap_recon_spike = ImageConstruct(recon_spikes,p,n,m)
    
    return new_spike,representations,overlap_recon_image,overlap_recon_spike,Patches,recon_spikes
            
def Reconstruct_Hidden(fmaps,p,units,w,thr):
    in_patches = list()
    D1 = len(fmaps)
    T=np.shape(fmaps[0])[2]
    dim1 = np.shape(fmaps[0])[0]
    dim2 = np.shape(fmaps[0])[1]
    
    for d in range(D1):
        patches,new_spikes = FeatureMap_Patch(fmaps[d],p,True,True)
        in_patches.append(patches)
        del patches
    L=len(in_patches[0])
    post_spikes = np.zeros([L,units,T])
    for l in range(L):
        for u in range(units):
            U = np.zeros([T])        
            for t in range(1,T):
                y_t = np.zeros([p*p*D1])
                for d1 in range(D1):
                    y_t[d1*p*p:p*p*(d1+1)] = in_patches[d1][l][:,:,t].flatten()
                if (np.sum(y_t)==0):
                    U[t] = U[t-1]*np.exp(-1)
                else:
                    U[t] = np.dot(w[:,u].T,y_t)/(np.linalg.norm(w[:,u])*np.linalg.norm(y_t)+0.000000000000000001)+(U[t-1]*np.exp(-1))
                    # This is for Test of DCNN
                    #U[t] = np.dot(w[:,u].T,y_t)/(np.linalg.norm(w[:,u]))+(U[t-1]*np.exp(-1))
                
                if (U[t]>=thr):
                    #print(U[t],t,u)
                    U[t]=0
                    post_spikes[l,u,t] = 1
    repre_spike = list()
    #print(np.shape(post_spikes))
    for u_map in range(units):
        #print(np.shape(post_spikes[:,u_map,:]))
        repre_spike.append(two_D_spike(post_spikes[:,u_map,:],dim1,dim2))
                    
    return repre_spike     

def flatten_3D(maps):
    n,m,T = np.shape(maps)
    new_map = np.zeros([n*m,T])
    for i in range(n):
        new_map[i*m:(i+1)*m,:] = maps[i,:,:]
    return new_map

        
def Flatter(featureMaps):
    R = len(featureMaps)
    D = len(featureMaps[0])
    n,m,T = np.shape(featureMaps[0][0])
    features = np.zeros([R,n*m*D,T])
    for r in range(R):
        for d in range(D):
            features[r,d*n*m:(d+1)*n*m,:] = flatten_3D(featureMaps[r][d])
    return features
    
def Flatter2D(featureMaps,level,T1,T):
    D = len(featureMaps[level])
    n,m = np.shape(featureMaps[level][0])
    features = np.zeros([n*m*D])
    for d in range(D):
        features[d*n*m:(d+1)*n*m] = featureMaps[level][d].flatten()
    return Poisson(features,T,float(T1))

class Convolution_Pooling_1(object):
        def __init__(self,image,w,p,stride,T,threshold):
                self._image = image
                self._w = w
                self._p = p
                self._stride = stride
                self._T = T
                n = np.shape(image)[0]
                m = np.shape(image)[1]
                self._threshold = threshold
                self._feature_maps=list()
                self._pooled_feature_maps = list()

                self._n2 = n-p+1
                self._m2 = m-p+1

                self._depth = np.shape(w)[1]

                self._patches = Image_Patch(self._image,self._p)
                self._patch_no = len(self._patches)

                self._u_tot = np.zeros([self._depth,self._patch_no])
                self._spikes = np.zeros([self._depth,self._patch_no,self._T])
                self.pool = mp.Pool()

        def _log_result_conv(self,result):
                # This is called whenever foo_pool(i) returns a result.
                # result_list is modified only by the main process, not the pool workers.
                #result_list.append(result)
                self._u_tot[result[2]:result[3],:] = result[1]
                self._spikes[result[2]:result[3],:,:] = result[0] 
        
        def _Conv_Sec(self,start,end):
                temp_self_spike = np.zeros([end-start,self._patch_no,self._T])
                temp_self_utot = np.zeros([end-start,self._patch_no])
                for i in range(start,end):
                        for j in range(self._patch_no):
                                U = np.zeros([self._T])
                                patchy = self._patches[j].flatten()
                                pre_spikes = Poisson(patchy,self._T)
                                for t in range(1,self._T):
                                        temp_spike = pre_spikes[:,t]
                                        U[t] = np.dot(self._w[:,i],temp_spike) + (U[t-1]*np.exp(-1))
                                        if (U[t]>=self._threshold):
                                                U[t]=0
                                                #self._spikes[i,j,t] = 1
                                                temp_self_spike[i,j,t] = 1
                                        #self._u_tot[i,j]+=np.dot(self._w[:,i],temp_spike)
                                        temp_self_utot[i,j]+=np.dot(self._w[:,i],temp_spike)
                
                return temp_self_spike,temp_self_utot,start,end
        
        def Conv2(self,cores):
                sec_size = int(self._depth/cores)
                if __name__ == 'SpikeConv':
                        print("Hello2")
                        #pool = mp.Pool()
                        for i in range(cores):
                            self.pool.apply_async(self._Conv_Sec, args = (i*sec_size, (i+1)*sec_size), callback = self._log_result_conv)
                        self.pool.close()
                        self.pool.join()
                        print("Hello")
                        #self._Conv_Sec(0,self._depth)

        def Pool2(self):
                self._Pooling()

                return self._feature_maps, self._pooled_feature_maps, self._spikes
                

        

        def _two_D(self,vector):
                index = 0
                new_map = np.zeros([self._n2,self._m2])
                for i in range(self._n2):
                        for j in range(self._m2):
                                new_map[i,j] = vector[i*self._m2+j]
                return new_map
        
        def _Pooling(self):
                for i in range(self._depth):
                        feuture_map_flatten = self._u_tot[i,:]
                        map_temp = self._two_D(feuture_map_flatten)
                        self._feature_maps.append(map_temp)

                        new_n2 = self._n2
                        new_m2 = self._m2

                        if (self._n2%self._stride!=0):
                                new_n2+=(self._stride-(self._n2%self._stride))
                        if (self._m2%self._stride!=0):
                                new_m2+=(self._stride-(self._m2%self._stride))
                        pooled_map_temp = np.zeros([int(new_n2/self._stride),int(new_m2/self._stride)])

                        temp_2d_map = np.zeros([new_n2,new_m2])
                        temp_2d_map[0:self._n2,0:self._m2]=map_temp
                        #print(temp_2d_map[0:10,0:10])
                        index_n = 0
                        
                        while (index_n<=new_n2-self._stride):
                                index_m = 0
                                while (index_m<=new_m2-self._stride):
                                        pooled_map_temp[int(index_n/self._stride),int(index_m/self._stride)]=np.max(
                                                temp_2d_map[index_n:index_n+self._stride,index_m:index_m+self._stride])
                                        index_m+=self._stride
                                index_n+=self._stride
                        
                        self._pooled_feature_maps.append(pooled_map_temp)
                                        
                                
                                
        
