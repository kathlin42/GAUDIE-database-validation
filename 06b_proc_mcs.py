# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 19:01:02 2022

@author: hirning
"""

import pandas as pd
import os
import helper_plotting as pb
import config_analysis

bids_root = os.path.join(config_analysis.project_root, "gaudie_audio_validation")

save_path = os.path.join(bids_root, 'derivatives')
for i_var, setting in enumerate(['valence_based', 'arousal_based', 'dominance_based']):
    df_save = pd.DataFrame()
    df = pd.read_csv(os.path.join(save_path, setting, 'Probability_errorous_labeling_'+ setting +'.csv'), header = 0, decimal='.', sep=';' )
    for audio in df['Audio'].unique():
        vals = df.loc[df['Audio'] == audio, 'Probability_errorous_labeling'].values
        dict_b = pb.bootstrapping(vals,
                                 numb_iterations = 4000,
                                 alpha =0.95,
                                 as_dict = True,
                                 func = 'mean')
        dict_b['ID'] = audio
        df_save = df_save.append(pd.DataFrame.from_dict(dict_b, orient ='index').T)
    for col in df_save.columns[:-1]:
        df_save[col] = pd.to_numeric(df_save[col])
    df_save.to_csv(os.path.join(save_path, setting, 'Averaged_probability_errorous_labeling_'+ setting +'.csv'), header = True, decimal=',', sep=';' )
