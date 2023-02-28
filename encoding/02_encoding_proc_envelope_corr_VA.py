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
import config_analysis

repository = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
audio_files = os.path.join(repository, 'derivatives', 'encoding', 'encoding_mel_hilb')
meta = os.path.join(audio_files, 'meta_data')
source_data = os.path.join(repository, 'source_data', 'behavioral')
save = os.path.join(audio_files, 'VA_correlation_analysis')

fs_ratings = 1 # in Hz
if not os.path.exists(save):
    os.makedirs(save)
    
subj_list = os.listdir(source_data)
subj_list = [sub for sub in subj_list if len(sub) == 3]
permutation = True

mean_zband_envelope_files = [file for file in os.listdir(audio_files) if ('z_score' in file) and ('pkl' in file)]
mean_band_envelope_files = [file for file in os.listdir(audio_files) if not ('z_score' in file) and ('pkl' in file)]

with open(os.path.join(meta,'dict_sr.pkl'), 'rb') as f:
    dict_sr = pickle.load(f)
with open(os.path.join(meta,'dict_freq.pkl'), 'rb') as f:
    dict_freq = pickle.load(f)

for permutation in [False, True]:
    for method, z_score  in zip([mean_zband_envelope_files, mean_band_envelope_files],['z_score', 'no_z_score']):
        dict_v = {}
        dict_a = {}
        print(z_score)
        for file in method:
            file_name = os.path.basename(file).split('.')[0][14:]
            if z_score == 'z_score':
                file_name = file_name[8:]
            print(file_name)
            with open(os.path.join(audio_files,file), 'rb') as f:
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
            with open(os.path.join(audio_files,file), 'rb') as f:
                mean_band_envelope = pickle.load(f)
            if permutation: 
                random.shuffle(dict_a[file_name])
                random.shuffle(dict_v[file_name])
                
            stats_a, p_a = stats.pearsonr(dict_a[file_name],mean_band_envelope)
            stats_v, p_v = stats.pearsonr(dict_v[file_name],mean_band_envelope)
            
                
            
            df_save = pd.DataFrame({'norm' :   [z_score],
                          'file' :   [file_name],
                          'pc_stats_a': [np.round(stats_a,3)],
                          'pc_p_a':     [np.round(p_a,3)],
                          'pc_stats_v': [np.round(stats_v,3)],
                          'pc_p_v':     [np.round(p_v,3)],
                          'permutation': [permutation]})  
            df_save.to_csv(os.path.join(save + "correlations_mel_hilb.csv"), header = (not os.path.exists(os.path.join(save + "correlations_mel_hilb.csv"))), mode = 'a', decimal = ',', sep = ';')
       