# -*- coding: utf-8 -*-
"""
@author: hirning
"""
# =============================================================================
# Analysis of Position Effects on SAM Ratings
# =============================================================================

# =============================================================================
# Import Packages
# =============================================================================
import os
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',family='Times New Roman', weight = 'bold')

from scipy import stats
import pickle
import config_analysis
# =============================================================================
# Load data
# =============================================================================
bids_root = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
ratings_file = os.path.join(bids_root, 'derivatives', 'ratings', "df_features.csv")
save_path = os.path.join(bids_root, 'derivatives', 'positional_effects')
if not os.path.exists(save_path):
    os.makedirs(save_path)

df = pd.read_csv(os.path.join(ratings_file), decimal=',', sep=';', header = 0)
# =============================================================================
# Create DF for Positional Information and save
# =============================================================================
if not os.path.exists(os.path.join(bids_root, 'source_data', 'df_position_per_subj.csv')):
    df_position = pd.DataFrame()
    for subj in os.listdir(os.path.join(bids_root, 'source_data','behavioral')):
        files = [file for file in os.listdir(os.path.join(bids_root, 'source_data','behavioral', subj)) if file != 'ratings.csv']
        df_subj = pd.DataFrame()
        positions = []
        for file in files:
            # Converting the time in seconds to a timestamp
            date_info = time.ctime(os.path.getmtime(os.path.join(bids_root, 'source_data','behavioral', subj, file)))
            dt_tp = datetime.datetime.strptime(date_info, "%c")
            positions.append(dt_tp.time())
            if file.split('_')[2] == 'positive':
                file_id = file.split('_')[4]
            else:
                file_id = file.split('_')[3]
            df_subj = df_subj.append(pd.DataFrame({'subj': [subj],
                                                           'SAM_scale' : [file.split('_')[-1][:-4]],
                                                           'ID' : [file.split('_')[2] + '_' + file_id],
                                                           'audio' : [file],
                                                           'year' : [dt_tp.year],
                                                           'month' : [dt_tp.month],
                                                           'day' : [dt_tp.day],
                                                           'time' : [str(dt_tp.time())],
                                                           'hour' : [dt_tp.hour],
                                                           'min' : [dt_tp.minute],
                                                           'sec': [dt_tp.second]}))
        df_subj['position'] = np.nan
        for pos, tp in enumerate(sorted(positions)):
            df_subj.loc[df_subj['time'] == str(tp), 'position'] = pos + 1
        df_position = df_position.append(df_subj)
    df_position.to_csv(os.path.join(bids_root, 'source_data', 'df_position_per_subj.csv'), header = True, index = False, decimal = ',', sep = ';')
else:
    df_position = pd.read_csv(os.path.join(bids_root, 'source_data', 'df_position_per_subj.csv'), decimal=',', sep=';', header = 0)

# =============================================================================
# Merge Position Info with Ratings
# =============================================================================
df['ID'] = df['condition'].values + '_' + [val[-6:-4] for val in df['audio'].values]
df_position = df_position.pivot(index=['subj', 'ID','year', 'month', 'day'], columns='SAM_scale', values='position').reset_index()
df_merge = df.merge(df_position, on = ['subj', 'ID'])

lst_color = ['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet']
dict_color = {'positive': lst_color[2],
              'neutral' : lst_color[0],
              'negative' : lst_color[1]}
fs = 14
#Per Condition
for condition in df_merge['condition'].unique():
    df_corr = df_merge.loc[df_merge['condition'] == condition]
    for SAM_scale in df_corr.columns[-2:]:
        if SAM_scale == 'Arousal':
            sam_vals = df_corr['arousal_mean'].astype(float).values
            pos_vals = df_corr[SAM_scale].values
        elif SAM_scale == 'Valence':
            sam_vals = df_corr['valence_mean'].astype(float).values
            pos_vals = df_corr[SAM_scale].values
        print(condition, SAM_scale, 'r:', stats.pearsonr(sam_vals, pos_vals)[0], 'p:', stats.pearsonr(sam_vals, pos_vals)[1])

#Overall
for SAM_scale in df_merge.columns[-2:]:
    if SAM_scale == 'Arousal':
        sam_vals = df_merge['arousal_mean'].astype(float).values
        pos_vals = df_merge[SAM_scale].values
    elif SAM_scale == 'Valence':
        sam_vals = df_merge['valence_mean'].astype(float).values
        pos_vals = df_merge[SAM_scale].values
    print('Over all conditions', SAM_scale, 'r:', stats.pearsonr(sam_vals, pos_vals)[0], 'p:', stats.pearsonr(sam_vals, pos_vals)[1])

fig = plt.figure(figsize=(15, 10), facecolor='white')
fig.suptitle('Correlations between SAM Scales and Position', fontsize=fs + 4, fontweight='bold')
for idx, SAM_scale in enumerate(df_merge.columns[-2:]):
    if SAM_scale == 'Arousal':
        sam_vals = df_merge['arousal_mean'].astype(float).values
        pos_vals = df_merge[SAM_scale].values
    elif SAM_scale == 'Valence':
        sam_vals = df_merge['valence_mean'].astype(float).values
        pos_vals = df_merge[SAM_scale].values
    r, p = stats.pearsonr(sam_vals, pos_vals)
    print('Over all conditions', SAM_scale, 'r:', r, 'p:', p)
    # plot result
    ax = fig.add_subplot(len(df_merge.columns[-2:]), 1, idx + 1)
    for condition in df_merge['condition'].unique():
        df_corr = df_merge.loc[df_merge['condition'] == condition]
        if SAM_scale == 'Arousal':
            cond_sam_vals = df_corr['arousal_mean'].astype(float).values
            cond_pos_vals = df_corr[SAM_scale].values
        elif SAM_scale == 'Valence':
            cond_sam_vals = df_corr['valence_mean'].astype(float).values
            cond_pos_vals = df_corr[SAM_scale].values
        ax.scatter(cond_pos_vals, cond_sam_vals, alpha=.5, c = dict_color[condition],label=condition)
    if idx == 0:
        ax.legend(fontsize=fs + 1)
    ax.set_xlabel('Position in the Experiment', fontsize=fs + 1, fontweight='bold')
    ax.set_ylabel(SAM_scale, fontsize=fs +1, fontweight='bold')
    ax.tick_params(labelsize=fs)
    ax.plot(np.unique(pos_vals), np.poly1d(np.polyfit(pos_vals, sam_vals, 1))(np.unique(pos_vals)), c = 'grey')
    ax.set_title(SAM_scale + ' (r = ' + str(np.round(r, 3)) + ', p = ' + str(np.round(p, 3)) + ')', fontsize=fs + 2, fontweight='bold')
fig.tight_layout()
plt.show()
fig.savefig(os.path.join(save_path, 'corr_SAM_position.svg'))
plt.close()

