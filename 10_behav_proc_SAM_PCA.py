# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:59:17 2021

@author: hirning
"""
import numpy as np
import itertools
import pandas as pd
import os
from sklearn.metrics import f1_score, accuracy_score
from fcmeans import FCM
from sklearn.decomposition import PCA
import pickle
from scipy.stats import spearmanr
import random
import matplotlib.pyplot as plt
import helper_proc as hp
import config_analysis

repository = os.path.join(config_analysis.project_root, "gaudie_audio_validation")

data_path = os.path.join(repository, 'derivatives', 'ratings')
setting = 'theory-based' 
save_path = os.path.join(repository, 'derivatives', 'pca', setting)
if not os.path.exists(save_path):
    os.makedirs(save_path)

df = pd.read_csv(os.path.join(data_path,'preprocessed_features.csv'), decimal=',', sep=';')
condition_list = df.loc[:, 'condition']
audio_list = df.loc[:, 'audio']
if setting == 'theory-based':
    drop = [col for col in df.columns if col not in ['valence_mean', 'arousal_mean', 'dominance']]    
elif (setting == 'full') or (setting == 'data-driven'):    
    drop = ['condition','audio', 'subj']
df = df.drop(drop, axis = 1)
df = df.apply(pd.to_numeric)

# =============================================================================
# PCA to 2 dimensions
# =============================================================================

df_pca = hp.PCA_features(df, df.columns, n_components = 3, save_path = save_path, plot_2_comp = False, plot_3_comp = False, plot_fcm_pca = False)
df_corr = pd.DataFrame(columns = ['PCA_Comp', 'Feature', 'r_s', 'p_s'])
for pca_col in df_pca.columns: 
    for feature in df.columns:
    #calculate Spearman Rank correlation and corresponding p-value
        rho, p = spearmanr(df[pca_col], df[feature])
        df_corr = df_corr.append(pd.DataFrame({'PCA_Comp':[pca_col],
                                               'Feature':[feature],
                                               'r_s':[rho],
                                               'p_s':[p],}))

df_corr['direction'] = ['negative' if row < 0 else 'positive' if row > 0 else 'zero' for row in df_corr.r_s]
df_corr.to_csv(os.path.join(save_path, "correlation_pca_comp_features.csv"), header = True, decimal = ',', sep = ';', index = False)

df_analysis = df_corr.copy()
df_analysis.r_s = abs(df_analysis.r_s)
df_analysis = df_analysis.sort_values(by='r_s', ascending = False)

for pca_col in df_pca.columns:
    print('#################')
    df_pca = df_analysis.loc[df_analysis['PCA_Comp'] == pca_col]
    df_pca = df_pca.loc[df_pca['r_s'] > 0.3]
    print(pca_col, df_pca.loc[:, ['Feature', 'direction']])
 