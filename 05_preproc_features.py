# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 11:23:25 2021

@author: hirning
"""

import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import config_analysis

# =============================================================================
# Load data 
# =============================================================================

repository = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
data_path = os.path.join(repository, 'derivatives', 'ratings')

df = pd.read_csv(os.path.join(data_path, 'df_features.csv'), decimal=',', sep=';')
df['emotion'] = ['positive' if emo in ['Freude', '�berraschung'] else 'negative' if emo in ['Trauer','Angst', 'Ekel', 'Wut'] else emo for emo in list(df['emotion'])]
gew_neg = ['Angst', 'Ekel', 'Hass', 'Wut', 'Ent�uschung', 'Trauer','Verachtung','Scham', 'Schuld', 'Bereuen']
gew_pos = ['Interesse', 'Freude', 'Vergn�gen', 'Zufriedenheit','Belustigung', 'Mitgef�hl', 'Stolz',  'Erleichterung']
df['geneva_wheel'] = ['positive' if emo in gew_pos else 'negative' if emo in gew_neg else emo for emo in list(df['emotion'])]
list_geneva_wheel = df['geneva_wheel']
df = pd.get_dummies(df, columns = ['emotion', 'geneva_wheel'])
for i, strength_response in enumerate(list_geneva_wheel):
    df.loc[i, 'geneva_wheel_' + strength_response] = df.loc[i, 'geneva_wheel_' + strength_response] *  df.loc[i, 'strength']
    
cols = [col for col in df.columns if col not in ['Unnamed: 0', 'strength']]
df = df.loc[:, cols].reset_index(drop = True)
df['audio'] = ['positive_' + row[-6:-4] if 'positive' in row else row[17:-4] for row in df['audio']]

num_cols = [col for col in df.columns if col not in ['condition', 'audio', 'subj']]

df.loc[:,num_cols] = df.loc[:,num_cols].apply(pd.to_numeric)
for col in num_cols:
    df[col] = (df[col] - df[col].mean()) / df[col].std()  
df.to_csv(os.path.join(data_path, 'preprocessed_features.csv'), index = False, header = True, decimal=',', sep=';' )
