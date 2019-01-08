#-*- coding: utf-8 -*-
import librosa
import librosa.display
import numpy as np
from dtw import dtw, show
from os.path import exists
from glob import glob
from platform import system as sys_chk

class recognition(object):
    #file_path = None   # string, path of input files
    #self.k = 0              # number of k
    #self.n = 0              # number of files
    #self.file_num = 0  # number of files in input directory
        
    def _process(self, x, y, path):
        
        # make y's length to x's length
        # x : ndarray, mfcc of wav, n1 by m
        # y : ndarray, mfcc of wav, n2 by m
        # path : tuple, dtw path
        
        yp = np.zeros(x.shape)
        
        for k in range(len(path[0])):
            ix, iy = path[0][k], path[1][k]        
            yp[ix] = y[iy]
            
        return yp

    def _processAll(self):
        x = None
        if self.debug:
                file = self.file_path+'0.wav'
                print(file,"is processed")
        if not exists(self.file_path+'0.npy'):
            wav1, sr = librosa.core.load(self.file_path+'0.wav')
            x = self._getMfcc(wav1,sr)            
            np.save(self.file_path+'0.npy',x)
            
        i = 1
        
        while exists(self.file_path+str(i)+'.wav'):
            if self.debug:
                file = self.file_path+str(i)+'.wav'
                print(file,"is processed")
            if exists(self.file_path+str(i)+'.npy'):
                i = i+1
                continue
        
            wav2, sr = librosa.core.load(self.file_path+str(i)+'.wav')
            y = self._getMfcc(wav2,sr)
            if x is None:
                x = np.load(self.file_path+'0.npy')
                
            d, _, _, path = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1))            
            yp = self._process(x,y,path)
            
            np.save(self.file_path+str(i)+'.npy',yp)
            i = i+1
        # save number of files
        self.n = i

    def _init_k_means(self):
        
        visit = np.zeros(self.n,dtype = np.bool)
        mfccs = None
        n_mfccs = []

        # initialize k mfcc
        for i in range(self.k):
            if i >= self.n:
                continue
            
            visit[i] = True
            file_name = self.file_path+str(i)+".npy"
            mfcc = np.load(file_name)
            if mfccs is None:
                mfccs = np.zeros((self.k,mfcc.shape[0],mfcc.shape[1]))

            mfccs[i] = mfcc
            n_mfccs = np.append(n_mfccs,1)

            
        for i in range(self.n):
            if visit[i]:
                continue
            file_name = self.file_path+str(i)+".npy"
            mfcc = np.load(file_name)

            dmin, jmin = np.inf, -1
            for j in range(self.k):            
                d, _, _, _ = dtw(mfccs[j], mfcc, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
                
                if d < dmin:
                    dmin = d
                    jmin = j
            
            j = jmin
            if self.debug:
                f = open(self.file_path + self.tag_name, 'r')
                label = f.readline()
                print(label,":",j,"is selected")
            mfccs[j] = mfccs[j]*n_mfccs[j]/(n_mfccs[j] + 1) + mfcc/(n_mfccs[j] + 1)
            n_mfccs[j] = n_mfccs[j] + 1 

       
        f = open(self.file_path + self.tag_name, 'r')
        label = f.readline()
        
        for i in range(len(mfccs)):
            self.mfccs[self._idx] = mfccs[i]
            self.labels[self._idx] = label
            self._idx = self._idx+1
        
    def loadData(self, k=2 ,file_path = None,debug = False):
        
        os_name = sys_chk()
     
        if os_name == 'Windows': 
            self.tag_name = "tag.txt"
        elif os_name == 'Linux':
            self.tag_name = "tag2.txt"
        else:
            self.tag_name = "tag.txt"
       
       
        self.debug = debug
        # k mean
        self.k = k
        # make k means npy file from each directory
        if file_path is None:
            file_path = './input/**/'
        else:
            file_path = file_path + "**/"
            
        file_paths = glob(file_path)
        self.mfccs= {}
        self.labels= {}
        self._idx = 0
        for file_path in file_paths:
            self.file_path = file_path
            # make same length of mfcc npy files from wav
            self._processAll()
            # make k means mfccs]
            self._init_k_means()
        
    def recognition(self,x):
        dmin, jmin = np.inf, -1
        for i in range(len(self.mfccs)):
            
            y = self.mfccs[i]
            
            d, _, _, path = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
            
            if d < dmin:
                dmin = d
                imin = i
                
        if self.debug:
            print(imin%self.k,"is selected")
        return self.labels[imin]
        
    def _getMfcc(self,wav,sr):
        mfcc = librosa.feature.mfcc(wav, sr, n_mfcc=13)
        mfcc = mfcc.T
        return mfcc
    
class validation(recognition):
    def __init__(self, train_path, test_path, debug = False):
        self.loadData(file_path = train_path)
        self.test_paths = glob(test_path+"**/")
        self.debug = debug

    def valid(self):
        
        score = 0
        total = 0

        for test_path in self.test_paths:
            f = open(test_path + self.tag_name, 'r')
            label = f.readline()
            f.close()
            
            wav_paths = glob(test_path+"*.wav")
            for wav_path in wav_paths:
                wav, sr = librosa.core.load(wav_path)
                x = self._getMfcc(wav,sr)
                xlabel = self.recognition(x)
                if xlabel == label:
                    score += 1
                total += 1
                if self.debug:
                    print("ans:",label[:-1], "guess:",xlabel[:-1])
                    print(score, total) 
        return score/total

if __name__ == '__main__':
    
    train_path = "./input/"
    test_path = "./test/"
    v1 = validation(train_path, test_path, debug = True)
    ret = v1.valid()
    print(ret*100,"%")
    
