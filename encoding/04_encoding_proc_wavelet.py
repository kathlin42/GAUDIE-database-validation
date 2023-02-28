# -*- coding: utf-8 -*-

#!/usr/bin/env python
import numpy as np
import glob
import os.path
import scipy.io.wavfile as wav
import pywt
os.chdir('..')
import config_analysis
if __name__ == "__main__":
    repository = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
    save_path = os.path.join(repository, 'derivatives', 'encoding', 'wavelet')
    wav_files = glob.glob(os.path.join(repository, 'stimulus_material', "*.wav"))
    if not os.path.exists("{}".format(save_path)):
        print('creating path for saving')
        os.makedirs("{}".format(save_path))

    save_path_meta = os.path.join(save_path, 'meta_data')
    if not os.path.exists("{}".format(save_path_meta)):
        print('creating path for saving meta_data')
        os.makedirs("{}".format(save_path_meta))    
    dict_sr = {}
    dict_freq = {}
    
    for wav_file in wav_files:
        print("Converting ", wav_file)
        
        # sig2, Fs = lbr.load(wav_file, sr=None, mono=True)
        rate, sig = wav.read(wav_file)
        if len(sig.shape) > 1:
               sig = np.mean(sig,axis=1) # convert a WAV from stereo to mono
        
        (cA, cD) = pywt.dwt(sig, 'db1') # Approximation and detail coefficients.
   
     
