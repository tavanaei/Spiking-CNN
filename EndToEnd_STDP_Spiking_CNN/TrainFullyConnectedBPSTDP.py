import numpy as np
from BPSTDP import SVP as svp
import cPickle as pickle
from RLSTDP import SpikeConv as cnv

np.random.seed(1)

NoConv = 2
levels = [16,32]

with open('RL8_Maps_'+str(levels[0])+str(levels[1])+'.pickle','rb') as f2:
    save2 = pickle.load(f2)
    dataSpike=save2['Maps']
    labelSpike = save2['Labels']
    
print("Data are loaded")
T1 = 20
T=50

outputThr = 1#[10,15,20,30,50,75] #50 #15 #Large threshold for 1000 and 1500 H
tau = 1

hiddenThr = [.1]
h = [1500]#[100,300,500,700,1000,1500]

LR=0.0002
alpha=7
dim = int(28/(np.power(2,NoConv)))

print(LR,h,hiddenThr,outputThr,alpha,levels[0],levels[1],NoConv)

w_h = [np.random.randn(dim*dim*levels[NoConv-1],h[0])*0.01]#,np.random.randn(h[0],h[1])]
for layer in range(1,len(h)):
    w_h.append(np.random.randn(h[layer-1],h[layer])*0.01)

w_o = np.random.randn(h[-1],10)*0.01
outRes = np.zeros([33000,10,T])
energies = np.zeros([33000])

for iteration in range(3):
    np.random.seed(iteration)
    for i in range(33000):
        if (i%1000==0): #5000 
            pred=np.zeros([1000])
            for i2 in range(1000): #10000
                spikes2 = cnv.Flatter2D(dataSpike[i2+33000],NoConv-1,T1,T)
                pred[i2],_,_ = svp.Test(spikes2,T,h,10,w_h,w_o,hiddenThr,outputThr)
            print(sum(pred==labelSpike[33000:34000].squeeze())/1000.0)

        w_h,w_o,outRes[i,:,:],energies[i] = svp.Train(cnv.Flatter2D(dataSpike[i],NoConv-1,T1,T),T,h,10,w_h,w_o,hiddenThr,outputThr,tau,int(labelSpike[i]),LR,alpha)
        #print(sum(outRes[i,:,:].T),labelSpike[i])

with open('SNN_W_'+str(h[0])+str(hiddenThr[0])+str(outputThr)+str(alpha)+str(levels[0])+str(levels[1])+str(NoConv)+'.pickle','w') as fw:
    save={'Wh': w_h,
          'Wo': w_o}
    pickle.dump(save,fw)

