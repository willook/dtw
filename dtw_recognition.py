import librosa
import librosa.display
import numpy as np
from dtw import dtw
import sounddevice as sd
import time




class recognition():       

    def loadData(self):
        with open('sounds/wavToTag.txt') as f:
            self.labels = np.array([l.replace('\n', '') for l in f.readlines()])
        #print("get labels complete")

        self.mfccs= {}
        for i in range(len(self.labels)):
            y, sr = librosa.load('sounds/{}.wav'.format(i))
            #print(labels[i]) 
            #sd.play(y,sr)
            #time.sleep(2)
            mfcc = librosa.feature.mfcc(y, sr, n_mfcc=13)
            self.mfccs[i] = mfcc.T
            
        #self.mfccs = np.load("./sounds/mfccs.npy")
        #print(self.mfccs.shape)
        
        
    def recognition(self,x):
        
        dmin, jmin = np.inf, -1
        for i in range(len(self.mfccs)):
            
            y = self.mfccs[i]
            
            d, _, _, _ = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
            if d < dmin:
                
                dmin = d
                imin = i
                #print(self.labels[imin],d,i)
            #print(self.labels[i],d)
        print(self.labels[imin])
        return self.labels[imin]
        
    def getMfcc(self,wav,sr):
        mfcc = librosa.feature.mfcc(wav, sr, n_mfcc=13)
        mfcc = mfcc.T
        return mfcc

   

