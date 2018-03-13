"""
@author: rhythm
"""
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
from math import log
from scipy import histogram, digitize, stats, mean, std
from collections import defaultdict
from sklearn.metrics.cluster import normalized_mutual_info_score
from numpy import convolve
####log calculation#####
log2= lambda x:log(x,2)

###Find 2^n that is equal to or greater than i.###
def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n 
    
############probability range##############

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    
    
############smoothing############## 
    
'''def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma'''
def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

#####variables declaration#####
#number of training files edit according to number of files you want to use for training
numTrainFiles=3
#loop counter
i=0
#frame size in seconds
frameSize = 0.04
# frame shift in seconds
frameShift = 10e-3
#training data path,comment if you want tou use less than 3 files for training and edit the path of your training files. 
trainPath1="path to training audio1.wav"
trainPath2="path to training audio2.wav"
trainPath3="path to training audio3.wav"
#test file path
testPath ="path to test audio.wav"
#svd array initialized to empty
B=[]
#threshold percentage to select
threshProp=0.7
#frames above threshold, init to empty
frames=[]
#number of frames above threshold
num=0
# output dir
pathdir="/home/rhythm/Desktop/seg_task/testOut"
#minimum number of frames to be continuous
numCounters=15

#######training data read and array creation#######
numColumns=5#number of SVD columns to use
for c in range(1, numTrainFiles+1):
    #svd array for each training file initialized to empty
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
B=B.reshape(numC,int((nfft1/2)+1))
#print B.shape

#########test data#########
fs, xtest = scipy.io.wavfile.read(testPath)
f, t, P= scipy.signal.spectrogram(xtest,fs,nfft=nfft1,nperseg=nperseg1,noverlap=noverlap1) 
P=np.asarray(P)
#print ('P shape',P.shape)
# number of frames in test file
numFrames = (P.shape)[1]
print ('numFrames',numFrames)

################calculate energy#############
F=np.dot(B,P)
maF=F.transpose()
A=[np.sqrt(sum(F[:,i]*F[:,i])) for i in range(numFrames)]
A=np.asarray(A)
#print ('Energy shape',A.shape)
A = (A - np.min(A))/(np.max(A)-np.min(A))

#########get frames above threshold###########
matEn = np.zeros((numFrames,1))
matMI = np.zeros((numFrames,1))
for ix in range(0,numFrames-1):
    #Mutual Information calculation
    matMI[ix,0]=normalized_mutual_info_score(maF[ix], softmax(maF[ix]))

for i in range(numFrames):
    if (matMI[i]>=threshProp):
       num=num+1
       frames.append(i)

#convert into array from list
frames=np.asarray(frames)
#print (frames)
print ('no of frames above threshold are ',num)

########test sound segmentation##########
x=1
an=0

#ch_wave and storing
arr=[]
for i in range(num):
     counter=0
     selectedFrame=[]     
     if((frames[x]-frames[x-1])==1): #to get frames that are continuous
            while((frames[x]-frames[x-1])==1) and (x<num-2):
                  selectedFrame.append(frames[x])
                  arr.append(frames[x])
                  counter=counter+1
                  x=x+1
                  #print x
     else:
          x=x+1
          #print x       	       
     #print (selectedFrame)
     if(counter>=numCounters):
        yMA = smooth(selectedFrame,1)
        yMA=np.asarray(yMA)
        #print yMA
        outpath=pathdir+'/part'+str(an)
        #calculate startTime using an=a+(n-1)d
        startTime=0+((yMA[0]-1)*frameShift)
        #calculate endTime using an=a+(n-1)d
        endTime=frameSize+((yMA[counter-1]-1)*frameShift)
        #segment and store bird calls
        os.system("ch_wave "+str(testPath)+" -o "+str(outpath)+".wav -start "+str(startTime)+" -end "+str(endTime))
        an=an+1
        #print startTime 
        #print endTime
        

