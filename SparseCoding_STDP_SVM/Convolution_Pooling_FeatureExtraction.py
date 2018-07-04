import matplotlib.pyplot as plt
import numpy as np
import SpikeConv as cnv
from six.moves import cPickle as pickle
import multiprocessing as mp
import time

depth = 32
T=20
patch_size = 5
R=3000
n=28
m=28

def ShowDigit(oneData):
    dimy=np.shape(oneData)[0]
    dim1=int(np.sqrt(dimy))
    tempPic=np.zeros([dim1,dim1])
    for i in range(0,dim1):
        for j in range(0,dim1):
            tempPic[i,j]=oneData[i*dim1+j]
    return tempPic

def ImageShow(im_2Ds,max_val):
    plt.figure()
    rows = int(np.sqrt(depth))
    for i in range(depth):
        plt.subplot(rows,rows,i+1)
        plt.imshow(im_2Ds[i,:,:]/max_val,vmin=0,vmax=1,cmap='Greys_r')
        plt.axis('off')
    plt.show()
    
def Store(file_name,data):
    with open(file_name,'wb') as f3:
        save4={'pooled_maps': data}
        pickle.dump(save4, f3, pickle.HIGHEST_PROTOCOL)


# Read Data from Pickle File containing Vectors of flatten images
data,label = cnv.Read_Data("Data_Test_MNIST.pickle")



with open("MNIST_V1_Final_32.pickle",'rb') as f:
    saved_file = pickle.load(f)
    v1_w = saved_file['Q']

patch_no = (n-patch_size+1)*(m-patch_size+1)
convolution_spikes = np.zeros([R,depth,patch_no,T])
#convolution_potential = np.zeros([depth,patch_no])

new_n2 = n-patch_size+1
new_m2 = m-patch_size+1
stride = 2
if ((n-patch_size+1)%stride!=0):
    new_n2+=(stride-((n-patch_size+1)%stride))
if ((m-patch_size+1)%stride!=0):
    new_m2+=(stride-((m-patch_size+1)%stride))

feature_maps = np.zeros([R,depth,n-patch_size+1,m-patch_size+1])
pooled_maps = np.zeros([R,depth,int(new_n2/stride),int(new_m2/stride),T])

#Ws=v1_w
Ws = cnv.Normalize(v1_w) #[patch**2 , depth]

print("***************Data Preparation is Done! ****************")

#a,b = cnv.Conv_Sec(digit,Ws[:,0],patch_size,2,T,5)

#start_time = time.time()

def log_result_convolution(result):
    convolution_spikes[result[4],result[3],:,:]=result[0]
    feature_maps[result[4],result[3],:,:]=result[1]
    pooled_maps[result[4],result[3],:,:,:]=result[2]

def Partial_Convolution(digit,index,digit_index):
    threshold = 1
    spike_trains,f_par_map,p_par_map = cnv.Conv_Sec(digit,Ws[:,index],patch_size,T,threshold,n-patch_size+1,m-patch_size+1,stride,new_n2,new_m2)    
    return spike_trains,f_par_map,p_par_map,index,digit_index

def Convolution_Pooling(digit,k):
    if __name__ == '__main__':
        ##start_time = time.time()
        pool = mp.Pool()    
        for i in range(depth):
            pool.apply_async(Partial_Convolution, args = (digit,i,k), callback = log_result_convolution)
        pool.close()
        pool.join()
        
        ##end_time = time.time()
        
        

        if ((k+1)%100==0):
            print("***************Convolution and Pooling are Done! ****************")
            print(k)

for r in range(R):
    #print('Image # '+str(r))
    digit = cnv.Image_Read(data[r+6000,:],28)
    
    Convolution_Pooling(digit,r)

#
for i in range(30):
   if (i<9):
       s_name = "0"+str(i+1)
   else:
       s_name = str(i+1)
   print(s_name)
   Store('CompleteData\\P23\\Pooled_Spike_Partial_'+s_name+'.pickle',pooled_maps[i*100:(i+1)*100,:,:,:,:])
   
   print("Time consumed is: "+str(end_time-start_time)+" s")

   This section is for image show 
r=1   
f_max = np.max(feature_maps[r,:,:,:])
p_max = np.max(sum(pooled_maps[r,:,:,:,:].T))
spike_counts = np.zeros([R,depth,n-patch_size+1,m-patch_size+1])


for i in range(depth):
   spike_counts[r,i,:,:]=ShowDigit(sum(convolution_spikes[r,i,:,:].T))

s_max = np.max(spike_counts[r,:,:,:])

ImageShow(feature_maps[r,:,:,:],f_max)
ImageShow(sum(pooled_maps[r,:,:,:,:].T).T,p_max)    
ImageShow(spike_counts[r,:,:,:],s_max)
    
