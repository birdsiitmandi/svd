import matplotlib
import pylab
import librosa
import librosa.display
import numpy as np
import scipy.io.wavfile
import scipy.io
from scipy.io.wavfile import read
from scipy import signal
import scipy
import os
import matplotlib.pyplot as plt 
import sys

#####default check#####
if (len(sys.argv)<>3):
	print("Usage:\n build svd.py pathdir segmentedAudiosDir.\n")
	sys.exit(-1)

###Find 2^n that is equal to or greater than i.###
def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

#####variables declaration#####
#number of training files edit according to number of files you want to use for training
numTrainFiles=3
#loop counter
i=0
#frame size in seconds
frameSize = 20e-3
# frame shift in seconds
frameShift = 10e-3
#training data path,comment if you want tou use less than 3 files for training and edit the path of your training files. 
trainPath1="/mnt/6EC3-B6CC/rhythm/Desktop/blackandyellow_grosbeak_GHNP/part_5.wav"
trainPath2="/home/rhythm/Desktop/seg_task/8.WAV"
trainPath3="/home/rhythm/Desktop/seg_task/smallhimwoodpeck.wav"
#test file path
testPath =sys.argv[1]
#svd array initialized to empty
B=[]
#threshold percentage to select
threshProp=1.2
#frames above threshold, init to empty
frames=[]
#number of frames above threshold
num=0
# output dir
pathdir=sys.argv[2]

#######training data read and array creation#######
numColumns=5#number of SVD columns to use
for c in range(1, numTrainFiles+1):
    svdarr=[]
    #var=('trainPath'+str(c)).replace("'", "")
    if c==1:
        var=trainPath1
    elif c==2:
          var=trainPath2
    elif c==3:
          var=trainPath3
    fs, x= scipy.io.wavfile.read(var)
    # nfft
    nfft1=nextpow2(int(frameSize*fs))
    #frame size in samples
    nperseg1 = int(frameSize*fs)
    #frame shift in samples
    noverlap1 = int(frameShift*fs)
    # compute spectrogram
    f, t, U= scipy.signal.spectrogram(x,fs,nfft=nfft1,nperseg=nperseg1,noverlap=noverlap1)
    #print ('traindata shape',U.shape)
    #D=librosa.power_to_db(U, ref=np.median) # needed??
    x_svd=np.linalg.svd(U)
    W=np.asarray(x_svd[:1])
    W=W.reshape(int((nfft1/2)+1),int((nfft1/2)+1))
    svdarr=W[:,:5]
    #print svdarr.shape
    B.append(svdarr)
    numC=numColumns*c
    #print c
    #print numC
B=np.asarray(B)
#print B.shape
#transpose of B
B=B.transpose()
#print B.shape
B=B.reshape(numC,int((nfft1/2)+1))
#print B.shape
#########test data#########
fs, xtest = scipy.io.wavfile.read(testPath)
f, t, P= scipy.signal.spectrogram(xtest,fs,nfft=nfft1,nperseg=nperseg1,noverlap=noverlap1) 
P=np.asarray(P)
#print ('P shape',P.shape)
# number of frames in test file
numTestFrames = (P.shape)[1]
print ('numtestframes',numTestFrames)

################calculate energy#############
F=np.abs(np.dot(B,P))
A=[np.sqrt(sum(F[:,i]*F[:,i])) for i in range(numTestFrames)]
A=np.asarray(A)

#print("Printing A")
#print(A[15000])
#print ('Energy shape',A.shape)
A = A / np.linalg.norm(A)
A=np.log(A)
plt.plot(A)
plt.show()
#print A

#########get frames above threshold###########
for i in range(numTestFrames):
    if A[i]>(threshProp*np.mean(A)):
       num=num+1
       frames.append(i)
       
#convert into array from list
frames=np.asarray(frames)
#print (frames)
print ('no of frames above threshold are ',num)

########test sound segmentation##########
#counters
x=-1
an=0
#ch_wave and storing
for i in range(num):
     counter=0
     selectedFrame=[]     
     if((frames[x]-frames[x-1])==1): #to get frames that are continuous
            while((frames[x]-frames[x-1])==1) and (x<num-2):
                  selectedFrame.append(frames[x])
                  counter=counter+1
                  x=x+1
                  #print x
     else:
          x=x+1
          #print x       	       
     #print (selectedFrame)
     if(counter>=10):
        selectedFrame=np.asarray(selectedFrame)
        outpath=pathdir+'/part'+str(an)
        #calculate startTime using an=a+(n-1)d
        startTime=0+((selectedFrame[0]-1)*frameShift)
        #calculate endTime using an=a+(n-1)d
        endTime=frameSize+((selectedFrame[counter-1]-1)*frameShift)
        #segment and store bird calls
        os.system("ch_wave "+str(testPath)+" -o "+str(outpath)+".wav -start "+str(startTime)+" -end "+str(endTime))
        an=an+1
        #print startTime 
        #print endTime
        

