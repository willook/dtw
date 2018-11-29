import pyaudio, wave
import keyboard, time
import numpy as np
import librosa
import sounddevice as sd
from dtw_recognition import recognition


class record(recognition):

    def __init__(self,audio_num = 0):
        
        self.top_db = 80
        self.audio_num = audio_num
        self.WAVE_OUTPUT_FILENAME = str(audio_num)+".wav"
        self.CHUNK = 1024 
        self.FORMAT = pyaudio.paInt16 
        self.CHANNELS = 1
        self.sr = 16000 
        
    def _init_record(self):                
        #dtw recognition
        self.loadData()
        self.p = pyaudio.PyAudio() 

        self.stream = self.p.open(format=self.FORMAT, 
               channels=self.CHANNELS, 
               rate=self.sr, 
               input=True, 
               frames_per_buffer=self.CHUNK) 

    def _recording(self):
        print("say anything! "*3)
        self.frames = []
        while True:
            if keyboard.is_pressed('w'):
                break
            data = self.stream.read(self.CHUNK) 
            self.frames.append(data)

        
        print("record is ended...")
        self.stream.stop_stream() 
        self.stream.close() 
        self.p.terminate()
        #print(self.frames[0:10])
        
        
        wf = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb') 
        wf.setnchannels(self.CHANNELS) 
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT)) 
        wf.setframerate(self.sr) 
        wf.writeframes(b''.join(self.frames)) 
        wf.close()

       

    def play(self):
        wav, sr = librosa.core.load(self.WAVE_OUTPUT_FILENAME,sr=self.sr)
        sd.play(wav,sr)
        
     
    def guess(self):
        wav, _ = librosa.core.load(self.WAVE_OUTPUT_FILENAME,sr=self.sr)
        wav, _ = librosa.effects.trim(wav, top_db=self.top_db)
        librosa.output.write_wav(self.WAVE_OUTPUT_FILENAME, wav, self.sr)
        print(">> save as", self.WAVE_OUTPUT_FILENAME)

        #dtw recognition
        x = self.getMfcc(wav,self.sr)
        res = self.recognition(x)
        
            
        self.audio_num = self.audio_num +1
        self.WAVE_OUTPUT_FILENAME = str(self.audio_num)+".wav"


    def run(self):
        print("[ press <Q> ] recording start")
        print("[ press <W> ] recording stop")
        print("[ press <E> ] play recording")
        print("[ press <R> ] guess")
        print("[ press <A> ] exit")
        
        try:
            while True:
                if keyboard.is_pressed('q'):
                    time.sleep(0.2)
                    self._init_record()
                    self._recording()
                if keyboard.is_pressed('e'):
                    time.sleep(0.2)
                    self.play()
                if keyboard.is_pressed('r'):
                    time.sleep(0.2)
                    self.guess()
                if keyboard.is_pressed('a'):
                    time.sleep(0.2)
                    break
                
        except KeyboardInterrupt:
            self.run()
            
if __name__ == '__main__':
    r1 = record()
    r1.run()
    


