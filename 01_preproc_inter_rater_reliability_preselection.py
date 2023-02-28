# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:21:53 2021

@author: hirning
"""
# =============================================================================
# Load Packages
# =============================================================================
import os
import numpy as np
import pandas as pd
import config_analysis
from scipy.stats import spearmanr
# =============================================================================
# Calulcate Interrater Reliability
# =============================================================================
project_directory = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
rater_data = os.path.join(project_directory, 'ressources', 'audio_stimuli', 'Ratings_preselection.csv')
df_ratings = pd.read_csv(rater_data, sep= ';', decimal= '.', header = 0)
#Percentage of Aggreement
df_ratings['Aggreement'] = df_ratings['Rating: Person 1'].values == df_ratings['Rating: Person 2'].values
percentage_aggreement = df_ratings['Aggreement'].value_counts()[True]/ len(df_ratings)
#Interclass correlation coefficient
r_spearman = spearmanr(df_ratings['Rating: Person 1'].values, df_ratings['Rating: Person 2'].values)

combined_rating = np.mean([df_ratings['Rating: Person 1'].values, df_ratings['Rating: Person 2'].values], axis = 0)
print('n_preselection', combined_rating.shape)
print('n_selection', combined_rating[combined_rating >= 6].shape)
print('average', np.mean(combined_rating[combined_rating >= 6]))
print('std', np.std(combined_rating[combined_rating >= 6]))
print('min', np.min(combined_rating[combined_rating >= 6]))
print('max', np.max(combined_rating[combined_rating >= 6]))
print('percentage_aggreement', percentage_aggreement)
print('r_spearman', r_spearman[0])
