# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 22:41:17 2021

@author: hirning
"""
# =============================================================================
# Import Packages
# =============================================================================

import os
import pandas as pd
import numpy as np
from scipy import stats
import itertools
import config_analysis
# =============================================================================
# Load and Prepare data
# =============================================================================

bids_root = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
df = pd.read_csv(os.path.join(bids_root, 'derivatives', 'ratings', 'df_features.csv'), decimal='.', sep=';')
df_demo = pd.read_csv(os.path.join(bids_root, 'derivatives', 'questionnaires', 'demographics_scores.csv'), decimal='.', sep=';')
dict_gender = dict(zip(df_demo['ID'].values, df_demo['gender'].values))
df['gender'] = np.nan
for subj in dict_gender.keys():
    df.loc[df['subj']== subj, 'gender'] = dict_gender[subj]

# =============================================================================
# Overall - Gender
# =============================================================================
print('#############################')
print('OVERALL - GENDER')
print('#############################')
for var in ['valence_mean', 'arousal_mean']:
    df_group = df.groupby(['gender', 'audio']).mean()[var].reset_index()
    df_female = df_group.loc[df_group['gender']== 'female', df_group.columns != 'gender'].sort_values(by='audio')
    df_male = df_group.loc[df_group['gender']== 'male', df_group.columns != 'gender'].sort_values(by='audio')
    assert (df_female.audio.values == df_male.audio.values).any()
    r, p = stats.pearsonr(df_female[var].values, df_male[var].values)
    print(var.split('_')[0].upper(), 'r =', np.round(r, 3), 'p =', np.round(p, 3))
    print('========================================================')
# =============================================================================
# Per Condition - Gender
# =============================================================================
print('#############################')
print('PER CONDITION - GENDER')
print('#############################')
for var in ['valence_mean', 'arousal_mean']:
    df_group = df.groupby(['gender', 'audio', 'condition']).mean()[var].reset_index()
    for con in df_group['condition'].unique():
        df_con = df_group.loc[df_group['condition'] == con]
        df_female = df_con.loc[
            df_con['gender'] == 'female', ~df_con.columns.isin(['gender', 'condition'])].sort_values(
            by='audio')
        df_male = df_con.loc[
            df_con['gender'] == 'male', ~df_con.columns.isin(['gender', 'condition'])].sort_values(
            by='audio')
        assert (df_female.audio.values == df_male.audio.values).any()
        r, p = stats.pearsonr(df_female[var].values, df_male[var].values)
        print(var.split('_')[0].upper(), con, 'r =', np.round(r, 3), 'p =', np.round(p, 3))
    print('========================================================')
# =============================================================================
# Per Condition - Within Gender
# =============================================================================
print('#############################')
print('PER CONDITION - WITHIN GENDER')
print('#############################')
for var in ['valence_mean', 'arousal_mean']:
    df_group = df.groupby(['gender', 'audio', 'condition', 'subj']).mean()[var].reset_index()
    for con in df_group['condition'].unique():
        df_con = df_group.loc[df_group['condition'] == con]
        for gender in df_con['gender'].unique():
            df_gender = df_con.loc[df_con['gender'] == gender]
            list_pairs = [[], []]
            for audio in df_gender.audio.unique():
                df_audio = df_gender.loc[df_gender['audio'] == audio]
                for comb in list(itertools.combinations(df_audio['subj'].values, 2)):
                    list_pairs[0].append(df_audio.loc[df_audio['subj'] == comb[0], var].values[0])
                    list_pairs[1].append(df_audio.loc[df_audio['subj'] == comb[1], var].values[0])
            assert (df_female.audio.values == df_male.audio.values).any()
            r, p = stats.pearsonr(list_pairs[0], list_pairs[1])
            print(var.split('_')[0].upper(), con, gender, 'r =', np.round(r, 3), 'p =', np.round(p, 3))
    print('========================================================')