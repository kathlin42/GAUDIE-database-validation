# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 21:17:46 2021

@author: hirning
"""

# =============================================================================
# Packages
# =============================================================================
import numpy as np
import glob
import os.path
import json
import re
import scipy.io.wavfile as wav
from scipy import stats
import librosa as lbr 

import matplotlib.pyplot as plt
from scipy.signal import hilbert
import librosa.display
import pickle

os.chdir('..')
import config_analysis

def get_mel_spectrogram(filename, log=True, sr=44100, hop_length=512, **kwargs):
#    '''Returns the (log) Mel spectrogram of a given wav file, the sampling rate
#    of that spectrogram and names of the frequencies in the Mel spectrogram
#
#    Parameters
#    ----------
#    filename : str, path to wav file to be converted
#    sr : int, sampling rate for wav file
#         if this differs from actual sampling rate in wav it will be resampled
#    log : bool, indicates if log mel spectrogram will be returned
#    kwargs : additional keyword arguments that will be
#             transferred to librosa's melspectrogram function
#
#    Returns
#    -------
#    a tuple consisting of the Melspectrogram of shape (time, mels), the 
#    repetition time in seconds, and the frequencies of the Mel filters in Hertz 
#    '''
    wav, _ = lbr.load(filename, sr=sr)
    melspecgrams = lbr.feature.melspectrogram(y=wav, sr=sr, hop_length=hop_length,
                                              **kwargs)
    if log:
        melspecgrams[np.isclose(melspecgrams, 0)] = np.finfo(melspecgrams.dtype).eps
        melspecgrams = np.log(melspecgrams)
    log_dict = {True: 'Log ', False: ''}
    freqs = lbr.core.mel_frequencies(
            **{param: kwargs[param] for param in ['n_mels', 'fmin', 'fmax', 'htk']
                if param in kwargs})
    freqs = ['{0:.0f} Hz ({1}Mel)'.format(freq, log_dict[log]) for freq in freqs]
    return melspecgrams.T, sr / hop_length, freqs

if __name__ == "__main__":

    repository = os.path.join(config_analysis.project_root, "gaudie_audio_validation")

    save_path = os.path.join(repository, 'derivatives', 'encoding', 'melspectrum')
    wav_files = glob.glob(os.path.join(repository, 'stimulus_material', "*.wav"))
    if not os.path.exists("{}".format(save_path)):
        print('creating path for saving')
        os.makedirs("{}".format(save_path))
   
    for wav_file in wav_files:
        print("Converting ",wav_file)
        
        # sig2, Fs = lbr.load(wav_file, sr=None, mono=True)
        rate, sig = wav.read(wav_file)
        if len(sig.shape) > 1:
               sig = np.mean(sig,axis=1) # convert a WAV from stereo to mono
        ## set parameters ##
        #rate        = 44100                     # sampling rate
        winlen      = int(np.rint(rate*0.025))  # 1102 Window length std 0.025s
        overlap     = 0
        hoplen      = winlen-overlap            # 661 hop_length
        nfft        = winlen    # standard is = winlen = 1102 ... winlen*2 = 2204 ... nfft = the FFT size. Default for speech processing is 512.
        nmel        = 32       # n = the number of cepstrum to return = the number of filters in the filterbank
        lowfreq     = 100       # lowest band edge of mel filters. In Hz
        highfreq    = 8000      # highest band edge of mel filters. In Hz
        noay        = 3         # subplot: y-dimension
        noax        = 1         # subplot: x-dimension
        # foursec     = int(np.rint(rate*4.0)) # 4 seconds
        # start_time  = 0
    
        config = {"n_fft":nfft, "sr":rate, "win_length":winlen, 
                  "hop_length":hoplen, "n_mels":nmel, "fmax":highfreq, 
                  "fmin":lowfreq}  
        melspec, sr_spec, freqs = get_mel_spectrogram(wav_file, **config)
        zscore_melspec= stats.zscore(melspec, axis=0)
        
        # =============================================================================
        # Plotting        
        # =============================================================================
        fig, ax = plt.subplots()
        S_dB = lbr.power_to_db(zscore_melspec.T, ref=np.max)
        img = lbr.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_spec, fmax=highfreq, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        fig.suptitle('Mel-frequency spectrogram - ' + os.path.basename(wav_file).split('.')[0][17:])
     
        plt.savefig(os.path.join(save_path, 'Melspectrogram_z_scored_'+ os.path.basename(wav_file).split('.')[0][17:] +'.svg'))
        plt.close()
        
        # =============================================================================
        # Plotting without z-score       
        # =============================================================================
        fig, ax = plt.subplots()
        S_dB = lbr.power_to_db(melspec.T, ref=np.max)
        img = lbr.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_spec, fmax=highfreq, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        fig.suptitle('Mel-frequency spectrogram - ' + os.path.basename(wav_file).split('.')[0][17:])
     
        plt.savefig(os.path.join(save_path, 'Melspectrogram_'+ os.path.basename(wav_file).split('.')[0][17:] +'.svg'))
        plt.close()