import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import SpikeConv as cnv
from six.moves import cPickle as pickle
import time

depth = 64
T=20
patch_size = 5
ex_weight =  np.random.random([patch_size*patch_size,depth])
in_weight = np.zeros([depth,depth])
threshold = np.ones([depth])*5

train_size = 30000
test_size = 100

def ShowDigit(oneData):
    dimy=np.shape(oneData)[0]
    dim1=int(np.sqrt(dimy))
    tempPic=np.zeros([dim1,dim1])
    for i in range(0,dim1):
        for j in range(0,dim1):
            tempPic[i,j]=oneData[i*dim1+j]
    return tempPic


def Make_Data(data,label,rows,p):
    digits = list()
    labels = list()
    patches = list()
    for r in range(rows):
        temp_digit = cnv.Image_Read(data[r,:],28)
        digits.append(temp_digit)
        labels.append(label[r])
        patches.append(cnv.Image_Patch(temp_digit,p))
        del temp_digit
    return digits,labels,patches

# Read Data from Pickle File containing Vectors of flatten images
data,label = cnv.Read_Data("Data.pickle")

plt.imshow(ShowDigit(255-data[11,:]),cmap='Greys_r')
plt.axis('off')
plt.show()


digits,labels,patches = Make_Data(data,label,train_size+test_size,patch_size)

print(np.shape(digits))
print(np.shape(labels))
print(np.shape(patches))

a_time = time.time()
for iteration in range(1):
    print("Iteration is: %d" %iteration)
    for i in range(train_size):
        if (i%50==0):
            print(str(i)+" images are done!")
        digit = digits[i]
        sub_patches = patches[i]
        for patchy in sub_patches:
            sample_patch = patchy.flatten()
            if (np.sum(sample_patch-np.mean(sample_patch))==0):
                continue
            sample_patch = (sample_patch-np.mean(sample_patch))/np.std(sample_patch)
            ex_weight,in_weight,threshold = cnv.Train_Convolution_Patch_Zylberg(sample_patch,ex_weight,in_weight,threshold,T,depth)
e_time=time.time()
print(e_time-a_time)

#with open("MNIST_V1_Final_64.pickle",'wb') as f2:
#    save2 = {'Q': ex_weight,
#             'W': in_weight,
#             'THR': threshold}
#    pickle.dump(save2, f2, pickle.HIGHEST_PROTOCOL)

#,cmap="Greys_r"
plt.figure()
for i1 in range(0,depth):
    plt.subplot(int(np.sqrt(depth)),int(np.sqrt(depth)),i1+1)
    temp = ex_weight[:,i1]
    plt.imshow(ShowDigit((temp-np.min(temp))/(np.max(temp)-np.min(temp))))
    plt.axis('off')
plt.show()



