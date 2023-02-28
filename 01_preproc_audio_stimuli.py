# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:21:53 2021

@author: hirning
"""
# =============================================================================
# Packages
# =============================================================================

import numpy as np
import os
from scipy.signal import fftconvolve
from scipy.special import rel_entr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import librosa
import soundfile as sf
from librosa import display
import pandas as pd
import seaborn as sns
import pickle
from scipy.stats.stats import pearsonr
import scipy.stats as stats
from pydub import AudioSegment
import config_analysis
# =============================================================================
# Function
# =============================================================================
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# =============================================================================
# Paths
# =============================================================================
project_directory = os.path.join(config_analysis.project_root, "gaudie_audio_validation")

stimuli_directory = os.path.join(project_directory, 'ressources/audio_stimuli/GAUDIE')
raw_directory = os.path.join(stimuli_directory, 'raw_audio')
save_directory = os.path.join(stimuli_directory, 'preproc_audio')
renamed_directory = os.path.join(save_directory, 'renamed_audio')
conditions = ['positive', 'negative', 'neutral', 'test'] 
db = -20
normalized_directory = os.path.join(save_directory, str(db) + 'db_normalized_audio')

df_describe = pd.DataFrame()
# =============================================================================
# Preprocessing 
# =============================================================================
for condition in conditions:
    for i, sound_file in enumerate(os.listdir(os.path.join(raw_directory, condition))):
        if i < 10:
            i_int = i
            i = '0' + str(i)
        else: 
            i_int = i
        print('Condition', condition, 'File ID', i)
        y, sr = librosa.load(os.path.join(raw_directory, condition, sound_file))
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)          
        save_path = os.path.join(renamed_directory, condition)
        if not os.path.exists("{}".format(save_path)):
            print("Creating renamed_directory")
            os.makedirs("{}".format(save_path))
            
        # Write out audio as 24bit PCM WAV
        sf.write(os.path.join(save_path, condition + '_' + str(i) + ".wav"), y, sr, subtype='PCM_24')
        
        # =============================================================================
        # Save ID, File name and Condition
        # =============================================================================
        df_describe = df_describe.append(pd.DataFrame({'ID': [str(i) + '_' + condition],'Name': [sound_file], 'Sample_rate': [sr], 'Len_in_sec': ['{:.3f}'.format(y.shape[0]/sr)], 'Samples': [y.shape[0]], 'Tempo': [tempo]}))
df_describe.to_csv(os.path.join(project_directory, 'ressources/audio_stimuli', "Description_GAUDIE_stimuli.csv"), sep=';', decimal = ',', index=False, header=True) 
       
# =============================================================================
# Normalize Sounds        
# =============================================================================        
for condition in conditions:
    for i, sound_file in enumerate(os.listdir(os.path.join(renamed_directory, condition))):
        sound = AudioSegment.from_file(os.path.join(renamed_directory, condition,sound_file))
        normalized_sound = match_target_amplitude(sound, db)
        save_path = os.path.join(normalized_directory, condition)
        if not os.path.exists("{}".format(save_path)):
            print("Creating normalized_directory")
            os.makedirs("{}".format(save_path))
        normalized_sound.export(os.path.join(save_path, 'normalized_' + str(db) + 'dB_' + condition + '_' + str(i) + ".wav"), format="wav")



           