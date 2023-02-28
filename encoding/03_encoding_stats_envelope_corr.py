# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 20:08:24 2021

@author: hirning
"""
import os
import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt

os.chdir('..')
import helper_plotting as pb
import config_analysis

repository = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
audio_files = os.path.join(repository, 'derivatives', 'encoding', 'encoding_mel_hilb')
data = os.path.join(audio_files, 'VA_correlation_analysis', 'correlations_mel_hilb.csv')


# =============================================================================
# Stats and Plots for Envelope Correlation
# =============================================================================
for norm in ('no_z_score', 'z_score'):
    print(norm)
    save_norm = os.path.join(audio_files, 'VA_correlation_analysis', norm)
    if not os.path.exists(save_norm):
        os.makedirs(save_norm)
        
    df = pd.read_csv(data, decimal=',', sep=';')
    df = df.loc[df.norm ==norm].reset_index(drop = True) 
    df = df.rename(columns={'pc_p_a':'P-Value Arousal Correlation','pc_p_v':'P-Value Valence Correlation', 
                            'pc_stats_a': 'Stats-Value Arousal', 'pc_stats_v': 'Stats-Value Valence'})
    for idx, row in df.iterrows():
        if 'positive' in row['file']:
            row['file'] = 'positive'
        elif 'negative' in row['file']:
            row['file'] = 'negative'
        elif 'neutral' in row['file']:
            row['file'] = 'neutral'
        if row['permutation'] == True:
            row['file'] = 'Chance'
        df.iloc[idx,:] = row
        
    pb.plot_boxes_errorbar(list(df['file'].unique()),
                         df,
                         'file',
                         ['P-Value Arousal Correlation', 'P-Value Valence Correlation'], 
                         boot = 'mean', 
                         boot_size = 5000,
                         title='Bootstrapped Mel-Hilbert - Rating Correlation Results ' + norm, 
                         lst_color=['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet', 'deeppink', 'cyan', 'green'], 
                         save_path = save_norm, 
                         fwr_correction = True,
                         contrasts = False)
    plt.savefig(os.path.join(save_norm, 'Bootstrapped_p_values_corr.svg'))
    plt.close()
    
    
    
    pb.plot_boxes_errorbar(list(df['file'].unique()),
                         df,
                         'file',
                         ['Stats-Value Arousal', 'Stats-Value Valence'], 
                         boot = 'mean', 
                         boot_size = 5000,
                         title='Bootstrapped Mel-Hilbert - Rating Correlation Results ' + norm, 
                         lst_color=['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet', 'deeppink', 'cyan', 'green'], 
                         save_path = save_norm, 
                         fwr_correction = True,
                         contrasts = True)
    plt.savefig(os.path.join(save_norm, 'Bootstrapped_stats_values_corr.svg'))
    plt.close()