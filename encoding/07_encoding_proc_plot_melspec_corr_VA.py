# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 20:43:22 2021

@author: hirning
"""

# -*- coding: utf-8 -*-

#!/usr/bin/env python
import numpy as np
import os.path
from scipy import stats
import pickle
import pandas as pd
from scipy import stats
from scipy.signal import fftconvolve
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

os.chdir('..')
import config_analysis

repository = os.path.join(config_analysis.project_root, "gaudie_audio_validation")

encoding_data = os.path.join(repository, 'derivatives', 'encoding_mel_hilb')
meta = os.path.join(repository, 'derivatives', 'encoding_mel_hilb', 'meta_data')
source_data = os.path.join(repository, 'source_data', 'behavioral')
save = os.path.join(encoding_data, 'VA_correlation_analysis', 'plots')
if not os.path.exists(save):
    os.makedirs(save)

    
subj_list = os.listdir(source_data)
subj_list = [sub for sub in subj_list if len(sub) == 3]
lst_color=['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet', 'darkturquoise']

mean_zband_envelope_files = [file for file in os.listdir(encoding_data) if ('z_score' in file) and ('pkl' in file)]
mean_band_envelope_files = [file for file in os.listdir(encoding_data) if not ('z_score' in file) and ('pkl' in file)]

with open(os.path.join(meta,'dict_sr.pkl'), 'rb') as f:
    dict_sr = pickle.load(f)
with open(os.path.join(meta,'dict_freq.pkl'), 'rb') as f:
    dict_freq = pickle.load(f)

for method, z_score  in zip([mean_zband_envelope_files, mean_band_envelope_files],['z_score', 'no_z_score']):
    dict_v = {}
    dict_a = {}
    print(z_score)
    for file in method:
        file_name = os.path.basename(file).split('.')[0][14:]
        if z_score == 'z_score':
            file_name = file_name[8:]
        print(file_name)
        with open(os.path.join(encoding_data,file), 'rb') as f:
            mean_band_envelope = pickle.load(f)
        for subj in subj_list:
            print(subj)
            #subj = '000'
            try: 
                valence_rating = [i for i in os.listdir(os.path.join(source_data, subj)) if ('Valence' in i) and (file_name in i)][0]
                arousal_rating = [ii for ii in os.listdir(os.path.join(source_data, subj)) if ('Arousal' in ii) and (file_name in ii)][0]
            
                v = pd.read_csv(os.path.join(source_data, subj, valence_rating), decimal='.', sep=';')
                a = pd.read_csv(os.path.join(source_data, subj, arousal_rating), decimal='.', sep=';')
                up_array = pd.DataFrame(np.linspace(0, v.playback_time.max(), mean_band_envelope.shape[0]), columns=['playback_time']) 
                up_v = pd.merge_asof(up_array, v, on= 'playback_time',direction='nearest')
                up_a = pd.merge_asof(up_array, a, on= 'playback_time',direction='nearest')
                up_v = up_v['slider_pos'].values 
                up_a = up_a['slider_pos'].values 
                #Valence
                if file_name not in list(dict_v.keys()):
                    dict_v[file_name] = up_v
                else:
                    dict_v[file_name] = np.vstack((dict_v[file_name],up_v))
                #Arousal
                if file_name not in list(dict_a.keys()):
                    dict_a[file_name] = up_a
                else:
                    dict_a[file_name] = np.vstack((dict_a[file_name],up_a))
            except:
                print('Not seen from subj')
    for key in list(dict_v.keys()):
        dict_v[key] = dict_v[key].mean(0)
        dict_a[key] = dict_a[key].mean(0)
    if z_score == 'z_score':
        for key in list(dict_v.keys()):
            dict_v[key] = (dict_v[key] - dict_v[key].mean()) / dict_v[key].std()
            dict_a[key] = (dict_a[key] - dict_a[key].mean()) / dict_a[key].std()
          
    for file in method:
        file_name = os.path.basename(file).split('.')[0][14:]
        if z_score == 'z_score':
            file_name = file_name[8:]
        print(file_name)
        with open(os.path.join(encoding_data,file), 'rb') as f:
            mean_band_envelope = pickle.load(f)
        
        
        
        fig = plt.figure(figsize=(15, 10), facecolor='white')   
        fig.suptitle('Correlation Audio - Ratings ' + file_name, fontsize = 20, fontweight='bold')
        plt.plot(mean_band_envelope, color = lst_color[0])
        plt.plot(dict_v[file_name], color = lst_color[1])
        plt.plot(dict_a[file_name], color = lst_color[2])
        plt.xlabel('Samples (Sampling Rate 40.02 Hz)', fontsize=12, fontweight='bold')
        plt.ylabel('Response/Amplitude of the Envelop ' + z_score, fontsize=12, fontweight='bold')
        plt.legend([Line2D([0], [0], color=lst_color[0], lw=3),  
                    Line2D([0], [0], color='white', lw=3),
                    Line2D([0], [0], color=lst_color[1], lw=3), 
                    Line2D([0], [0], color=lst_color[2], lw=3)], 
                  ['Average Envelope over Bands', 
                   '',
                   'Average Valence Ratings over Subjs',
                   'Average Arousal Ratings over Subjs'], 
                  loc ='upper right', 
                  facecolor='white',
                  fontsize='x-large',
                  ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(save, 'Line_plot_'+ file_name +'.svg'))
        plt.close()