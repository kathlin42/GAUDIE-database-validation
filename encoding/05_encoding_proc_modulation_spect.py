# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:23:47 2021

@author: hirning
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa as lbr
import json
import os
import warnings            
import pandas as pd 
import glob
os.chdir('..')
import config_analysis

repository = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
save_path = os.path.join(repository, 'derivatives', 'encoding', 'modulation_spectrum')
wav_files = glob.glob(os.path.join(repository, 'stimulus_material', "*.wav"))
if not os.path.exists("{}".format(save_path)):
    print('creating path for saving')
    os.makedirs("{}".format(save_path))

sr = 44100
n_fft = 882
hop_length = 882
mps_n_fft = 100 
mps_hop_length = 100
n_mels = 64
plot_mps = True
for file in os.listdir(wav_files):
    #file = os.listdir(wav_files)[0]
    # =============================================================================
    # Read Data
    # =============================================================================
    wav, _ = lbr.load(os.path.join(repository, 'stimulus_material', file), sr = sr)
    
    # =============================================================================
    # Compute the RMS     
    # =============================================================================
    '''
    Compute root-mean-square (RMS) value for each frame, either from the audio samples y or from a spectrogram S.
    Computing the RMS value from audio samples is faster as it doesn’t require a STFT calculation. 
    However, using a spectrogram will give a more accurate representation of energy over time because its frames can be windowed, 
    thus prefer using S if it’s already available.
    '''

    S, phase = lbr.magphase(lbr.stft(wav))       
    rms = lbr.feature.rms(S=S)
    
    # =============================================================================
    # Compute Mel Spectrogram
    # =============================================================================
        
    mel_spec = lbr.feature.melspectrogram(y=wav, sr=sr, hop_length=hop_length)
    
    # Transpose Mel spectrogram for further analyses and compatibility
    mel_spec = mel_spec.T


    # =============================================================================
    # Sanity Check
    # =============================================================================
    if mps_n_fft >= mel_spec.shape[0]:
        raise ValueError("The mps window size exceeds the Mel spectrogram. Please enter a smaller integer.")
    
    if mps_hop_length >= mel_spec.shape[0]:
        raise ValueError("The mps step size exceeds the Mel spectrogram. Please enter a smaller integer.")
        
    # =============================================================================
    # Step 4 
    # =============================================================================

    '''
    Extract MPS by looping through spectrogram with pre-set window size (mps_n_fft) and pre-set hop_length (mps_hop_length). 
    Also extracting the Nyquist Frequency. 
    mps_all will be converted to a numpy array.
    '''
    mps_all = []
    mps_plot = []
    nyquist_mps = int(np.ceil(mel_spec.shape[1]/2))
    
    for i in range(1,int(np.floor(mel_spec.shape[0] / mps_n_fft))+1):
    
        #Extract mps for predefined window
        mps = np.fft.fft2(mel_spec[mps_n_fft*(i-1):mps_n_fft*i,:])
       
        # use absoulte and shifted frequencies
        mps = np.abs(np.fft.fftshift(mps))
        
        # only take quarter of complete MPS (due to mirroring)
        mps = mps[int(mps_n_fft/2):,nyquist_mps:]
        
        # Define variable for later plotting
        mps_plot.append(mps)
       
        # Flattening the mps to a vector
        mps = np.reshape(mps,(1,np.size(mps)))
        
        # Append mps to mps all
        mps_all.append(mps)
        
    # Convert mps_all into an array outside the loop
    mps_all = np.array(mps_all)
    
    # Convert mps_plot into an array outside loop
    mps_plot = np.array(mps_plot)
    
    # Concatinating the MPS row-wise
    mps_all = np.concatenate(mps_all)

# =============================================================================
# Step 5 Extract Axes Labels
# =============================================================================

    # Calculate the raw signal length
    raw_length_sec = (len(wav)/sr)
    raw_length_min = raw_length_sec/60
    
    # Sampling Rate in Mel Spectrogram
    fs_spectrogram = round(len(mel_spec)/(raw_length_sec))#if i roiund it the fs_spec will be 0 
    
    # Sampling rate in MPS 
    fs_mps = round(mps_n_fft/(raw_length_min))
    
    # Extract Axes units for plotting 
    # Calculate step sizes for MPS based on the logarithmic frequencies
    mel_freqs = lbr.mel_frequencies(n_mels = n_mels)
    freq_step_log = np.log(mel_freqs[2]) - np.log(mel_freqs[1])
    
    # Calculate labels for X and Y axes
    mps_freqs = np.fft.fftshift(np.fft.fftfreq(mel_spec.shape[1], d = freq_step_log)) # returns fourier transformed freuqencies which are already shifted (lower freq in center))
    mps_times = np.fft.fftshift(np.fft.fftfreq(mps_n_fft, d = 1. /fs_spectrogram)) 


    # =============================================================================
    # Step 6.
    # =============================================================================
    
    '''
    Plot Mel Spectrogram of first window and according MPS next to each other
    '''
    if plot_mps:
        fig, (ax1,ax2)= plt.subplots(1, 2, figsize=(20, 10))
        
        # use only first window of Mel spectrogram to plot
        first_mel = mel_spec[0:mps_n_fft,:]
        
        #extract time and frequency axes
        time = np.arange(0,mps_n_fft)*fs_spectrogram
        frequency = np.arange(0,mel_spec.shape[1])*fs_mps
        
        # define first plot (Mel spectrgram)
        image1 = ax1.imshow(first_mel.T, origin = 'lower', aspect = 'auto')
        ax1.set_xticks(np.arange(0,mps_n_fft,20))
        ax1.set_yticks(np.arange(0,first_mel.shape[1],10))
        x1= ax1.get_xticks()
        y1= ax1.get_yticks()
        ax1.set_xticklabels(['{:.0f}'.format(xtick) for xtick in time[x1]])
        ax1.set_yticklabels(['{:.2f}'.format(ytick) for ytick in frequency[y1]])
        ax1.set_title('Mel Spectrogram 1st window')
        ax1.set_ylabel('Frequencyband (Hz)')
        ax1.set_xlabel('Time (s)')
        cbar = fig.colorbar(image1, ax = ax1, format='%+2.0f dB')
        cbar.set_label('dB')
        
        # define second plot (MPS for Mel spectrogram first window)
        image2 = ax2.imshow(np.log(mps_plot[0,:,:].T), origin = 'lower', aspect = 'auto')
        
        # use only half of the frequqecies (up to Niquist so the MPS is not mirrored)
        mps_freqs2 = mps_freqs[nyquist_mps:,]
        
        # use only the right side off the mirrored Y axis 
        mps_times2 = mps_times[int(mps_n_fft/2):,]
        
        ax2.set_xticks(np.arange(0,len(mps_times2),20))
        ax2.set_yticks(np.arange(0,len(mps_freqs2),8))
        x2= ax2.get_xticks()
        y2= ax2.get_yticks()
        ax2.set_xticklabels(['{:.0f}'.format(xtick2) for xtick2 in mps_times2[x2]])
        ax2.set_yticklabels(['{:.2f}'.format(ytick2) for ytick2 in mps_freqs2[y2]])
        ax2.set_title(' MPS for Mel Spectrogram (1st window)')
        ax2.set_xlabel('Temporal Modulation (mod/s)')
        ax2.set_ylabel('Spectral Modulation (cyc/oct)')
        cbar = fig.colorbar(image2, ax=ax2)
        cbar.set_label('(log) MPS')
        plt.savefig(os.path.join(save_path, file[17:-4] + '.svg'))
        plt.close() 
# =============================================================================
# Extract names of features in the MPS
# =============================================================================
    names_features = ['{0:.2f} mod/s {1:.2f} cyc/oct)'.format(mps_time, mps_freq) 
                  for mps_time in mps_times for mps_freq in mps_freqs]