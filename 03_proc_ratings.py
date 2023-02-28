# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 22:41:17 2021

@author: hirning
"""

# =============================================================================
# Analysis of the Ratings 
# =============================================================================

# =============================================================================
# Import Packages
# =============================================================================

import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rc('font',family='Times New Roman', weight = 'bold')
fs = 20
import pickle
import config_analysis
# =============================================================================
# Load data
# =============================================================================
bids_root = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
source_data = os.path.join(bids_root, 'source_data', 'behavioral')
save = os.path.join(bids_root, 'derivatives', 'ratings')
save_subj = 'per_subj'
if not os.path.exists(save):
    os.makedirs(save)
    
if not os.path.exists(os.path.join(save, save_subj)):
    os.makedirs(os.path.join(save, save_subj))
    
subj_list = os.listdir(source_data)
subj_list = [sub for sub in subj_list if len(sub) == 3]
dict_valence = {}
dict_arousal = {}
lst_color = ['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet']
dict_color = {'positive': lst_color[2],
              'neutral' : lst_color[0],
              'negative' : lst_color[1]}
if os.path.exists(os.path.join(save,'dict_valence.pickle')):
    with open(os.path.join(save,'dict_valence.pickle'), 'rb') as file:
        dict_valence = pickle.load(file)
    with open(os.path.join(save,'dict_arousal.pickle'), 'rb') as file:
        dict_arousal = pickle.load(file)
else:
    for subj in subj_list:
        #subj = '025'
        valence_ratings = [file for file in os.listdir(os.path.join(source_data, subj)) if 'Valence' in file]
        arousal_ratings = [file for file in os.listdir(os.path.join(source_data, subj)) if 'Arousal' in file]
        other_ratings = [file for file in os.listdir(os.path.join(source_data, subj)) if 'ratings' in file]

        # =============================================================================
        # Analyse Valence
        # =============================================================================
        fig, axes = plt.subplots(round(len(valence_ratings)/5),5, figsize=(25,15), sharex=False, sharey=True)
        axes = axes.flatten()
        list_average_valence_pos = []
        list_average_valence_neu = []
        list_average_valence_neg = []

        for file, ax in zip(valence_ratings, axes):
            scale = 'Valence'
            if 'negative' in file:
                condition = 'NEGATIVE'
                color = dict_color['negative']
                file_id = file[26:28]
            elif 'positive' in file:
                condition = 'POSITIVE'
                color = dict_color['positive']
                file_id = file[32:34]
            elif 'neutral' in file:
                condition = 'NEUTRAL'
                color = dict_color['neutral']
                file_id = file[25:27]

            df = pd.read_csv(os.path.join(source_data, subj, file), decimal='.', sep=';')
            if file not in list(dict_valence.keys()):
                dict_valence[file] = np.array(df['slider_pos'])
            else:
                try:
                    dict_valence[file] = np.vstack((dict_valence[file],np.array(df['slider_pos'])))
                except:
                    original_length = np.array(df['slider_pos']).shape[0]
                    if dict_valence[file].shape[-1] < original_length:
                        imputed_array = np.array(df['slider_pos'])[:dict_valence[file].shape[-1]]
                    else:
                        imputed_array = np.array(df['slider_pos'])
                        while dict_valence[file].shape[-1] != original_length:
                            imputed_array = np.append(imputed_array, df['slider_pos'].iloc[-1:])
                            original_length = original_length + 1
                    dict_valence[file] = np.vstack((dict_valence[file],imputed_array))

            if condition == 'NEGATIVE':
                list_average_valence_neg.append(np.mean(df['slider_pos'][30:]))
            if condition == 'POSITIVE':
                list_average_valence_pos.append(np.mean(df['slider_pos'][30:]))
            if condition == 'NEUTRAL':
                list_average_valence_neu.append(np.mean(df['slider_pos'][30:]))
            ax.plot(df['playback_time'], df['slider_pos'], color = color)
            ax.set_xlabel('Time in sec', fontsize=fs, fontweight = 'bold')
            ax.set_ylabel('SAM Rating with Scale 1 - 9', fontsize=fs, fontweight = 'bold')
            ax.set_title(condition + ' ' + file_id + ' ($M$=' + str(np.round(np.mean(df['slider_pos']),3)) + '+/-' + str(np.round(np.std(df['slider_pos']),3)) + ')', fontsize=fs)
            ax.set_ylim(0, 100)

        fig.suptitle(scale + ' Rating Participant ' + subj, fontsize=fs + 6, fontweight = 'bold')
        fig.legend([Line2D([0], [0], color=dict_color['negative'], lw=3),
              Line2D([0], [0], color=dict_color['neutral'], lw=3),
              Line2D([0], [0], color=dict_color['positive'], lw=3)],
              ['negative (NEG)', 'neutral (NEU)','positive (pos)'] ,fontsize='xx-large')

        # set the spacing between subplots
        plt.tight_layout(rect = [0, 0, 1, 0.96],h_pad=4)

        # save the file
        fig.savefig(os.path.join(save, save_subj, scale + "_ratings_" + subj + ".svg"))
        plt.close(fig)
        del fig, axes

        # =============================================================================
        # Analyse Arousal
        # =============================================================================
        fig, axes = plt.subplots(round(len(arousal_ratings)/5),5, figsize=(25,20), sharex=False, sharey=False)
        axes = axes.flatten()
        list_average_arousal_pos = []
        list_average_arousal_neu = []
        list_average_arousal_neg = []
        for file, ax in zip(arousal_ratings, axes):
            scale = 'Arousal'
            if 'negative' in file:
                condition = 'NEGATIVE'
                color = dict_color['negative']
                file_id = file[26:28]
            elif 'positive' in file:
                condition = 'POSITIVE'
                color = dict_color['positive']
                file_id = file[32:34]
            elif 'neutral' in file:
                condition = 'NEUTRAL'
                color = dict_color['neutral']
                file_id = file[25:27]
            df = pd.read_csv(os.path.join(source_data, subj, file), decimal='.', sep=';')
            ax.plot(df['playback_time'], df['slider_pos'], color = color)
            ax.set_xlabel('Time in sec', fontsize=fs, fontweight = 'bold')
            ax.set_ylabel('SAM Rating with Scale 1 - 9', fontsize=fs, fontweight = 'bold')
            ax.set_title(condition + ' ' + file_id + ' ($M$=' + str(np.round(np.mean(df['slider_pos']),2)) + '+/-' + str(np.round(np.std(df['slider_pos']),2)) + ')', fontsize=fs)
            ax.set_ylim(0, 100)

            if file not in list(dict_arousal.keys()):
                dict_arousal[file] = np.array(df['slider_pos'])
            else:
                try:
                    dict_arousal[file] = np.vstack((dict_arousal[file],np.array(df['slider_pos'])))
                except:
                    original_length = np.array(df['slider_pos']).shape[0]
                    if dict_arousal[file].shape[-1] < original_length:
                        imputed_array = np.array(df['slider_pos'])[:dict_arousal[file].shape[-1]]
                    else:
                        imputed_array = np.array(df['slider_pos'])
                        while dict_arousal[file].shape[-1] != original_length:
                            imputed_array = np.append(imputed_array, df['slider_pos'].iloc[-1:])
                            original_length = original_length + 1
                    dict_arousal[file] = np.vstack((dict_arousal[file],imputed_array))


            if condition == 'NEGATIVE':
                list_average_arousal_neg.append(np.mean(df['slider_pos'][30:]))
            if condition == 'POSITIVE':
                list_average_arousal_pos.append(np.mean(df['slider_pos'][30:]))
            if condition == 'NEUTRAL':
                list_average_arousal_neu.append(np.mean(df['slider_pos'][30:]))
        fig.suptitle(scale + ' Rating Participant ' + subj, fontsize=fs + 6, fontweight = 'bold')
        fig.legend([Line2D([0], [0], color=dict_color['negative'], lw=3),
              Line2D([0], [0], color=dict_color['neutral'], lw=3),
              Line2D([0], [0], color=dict_color['positive'], lw=3)],
              ['negative (NEG)', 'neutral (NEU)','positive (POS)'],fontsize='xx-large')

        # set the spacing between subplots
        plt.tight_layout(rect = [0, 0, 1, 0.96],h_pad=4)

        # save the file
        fig.savefig(os.path.join(save, save_subj, scale + "_ratings_" + subj + ".svg"))
        plt.close(fig)

        df_results_cont = pd.DataFrame.from_dict({'ID': subj,
                      'mean_valence_neg': np.round(np.mean(np.array(list_average_valence_neg)),3),
                      'std_valence_neg': np.round(np.std(np.array(list_average_valence_neg)),3),
                      'mean_valence_neu': np.round(np.mean(np.array(list_average_valence_neu)),3),
                      'std_valence_neu': np.round(np.std(np.array(list_average_valence_neu)),3),
                      'mean_valence_pos': np.round(np.mean(np.array(list_average_valence_pos)),3),
                      'std_valence_pos': np.round(np.std(np.array(list_average_valence_pos)),3),
                      'mean_arousal_neg': np.round(np.mean(np.array(list_average_arousal_neg)),3),
                      'std_arousal_neg': np.round(np.std(np.array(list_average_arousal_neg)),3),
                      'mean_arousal_neu': np.round(np.mean(np.array(list_average_arousal_neu)),3),
                      'std_arousal_neu': np.round(np.std(np.array(list_average_arousal_neu)),3),
                      'mean_arousal_pos': np.round(np.mean(np.array(list_average_arousal_pos)),3),
                      'std_arousal_pos': np.round(np.std(np.array(list_average_arousal_pos)),3)}, orient='index')

        df_results_cont.T.to_csv(os.path.join(save, "continuous_ratings.csv"), header = (not os.path.exists(os.path.join(save, "continuous_ratings.csv"))), mode = 'a', decimal = ',', sep = ';')
        # =============================================================================
        # Analyse Other Ratings
        # =============================================================================

        df = pd.read_csv(os.path.join(source_data, subj, other_ratings[0]), decimal='.', sep=';', header = 0)
        df = df.loc[(df['audio'] != 'audio') & (df['audio'] != 'Testaudio.wav') &(df['audio'] !='normalized_-20dB_test_00.wav')]
        lst_condition = []
        list_id = []
        for index, row in df.iterrows():
            if 'negative' in row['audio']:
                lst_condition.append('negative')
                list_id.append(row['audio'][26:28])
            elif 'positive' in row['audio']:
                lst_condition.append('positive')
                list_id.append(row['audio'][32:34])
            elif'neutral' in row['audio']:
                lst_condition.append('neutral')
                list_id.append(row['audio'][25:27])
        df['condition'] = lst_condition
        df['list_id'] = list_id
        numeric = ['geneva_slider','familiarity','dominance']
        df[numeric] = df[numeric].apply(pd.to_numeric, errors='coerce')

        df_negative = df.loc[df['condition']=='negative', :]
        df_positive = df.loc[df['condition']=='positive', :]
        df_neutral = df.loc[df['condition']=='neutral', :]

        df_negative.describe()
        df_positive.describe()
        df_neutral.describe()

        # =============================================================================
        # GENEVA WHEEL
        # =============================================================================
        lst_color_order = [dict_color['negative'], dict_color['neutral'], dict_color['positive']]
        df_gew = df.groupby(['condition', 'emotion']).count().reset_index().iloc[:,:3]
        df_gew['strength'] = df.groupby(['condition', 'emotion']).mean().reset_index()['geneva_slider']
        fig, axes = plt.subplots(len(df_gew['condition'].unique()),1, figsize=(12,10), sharex=False, sharey=False)
        axes = axes.flatten()
        for id_col, (ax, cond) in enumerate(zip(axes, sorted(list(df_gew['condition'].unique())))):
            df_plot = df_gew.loc[df_gew['condition']==cond]
            ax.bar(df_plot['emotion'],df_plot['audio'], color = lst_color_order[id_col])
            ax.set_title(cond + ' Condition')
            ax.set(xlabel='Emotions', ylabel= 'Frequency')
            ax.set_ylim(0, 8)

        fig.suptitle('Geneva Wheel Ratings Subject ID' + subj)
        fig.legend([Line2D([0], [0], color=dict_color['negative'], lw=3),
              Line2D([0], [0], color=dict_color['neutral'], lw=3),
              Line2D([0], [0], color=dict_color['positive'], lw=3)],
              ['negative', 'neutral','positive'],fontsize='xx-large')
        plt.show()
        # set the spacing between subplots
        plt.tight_layout()

        # save the file
        fig.savefig(os.path.join(save, save_subj, "GEW_" + subj + ".svg"))
        plt.close(fig)
        # =============================================================================
        # Familiarity
        # =============================================================================
        df_fam = df.groupby(['condition', 'audio']).mean().reset_index()
        df_fam = df_fam.loc[:,['condition', 'audio', 'familiarity']]
        fig, axes = plt.subplots(len(df_fam['condition'].unique()),1, figsize=(12,10), sharex=False, sharey=False)
        axes = axes.flatten()
        for id_col, (ax, cond) in enumerate(zip(axes, sorted(list(df_fam['condition'].unique())))):
            df_plot = df_fam.loc[df_fam['condition']==cond]

            ax.bar(df_plot['audio'],df_plot['familiarity'], color = lst_color_order[id_col])
            ax.set_title(cond + ' Condition')
            ax.set(xlabel='IDs', ylabel= 'Frequency')
            ax.set_ylim(0, 100)

        fig.suptitle('Familiarity Ratings Subject ID' + subj)
        fig.legend([Line2D([0], [0], color=dict_color['negative'], lw=3),
              Line2D([0], [0], color=dict_color['neutral'], lw=3),
              Line2D([0], [0], color=dict_color['positive'], lw=3)],
              ['negative', 'neutral','positive'],fontsize='xx-large')

        plt.show()
        # set the spacing between subplots
        plt.tight_layout()

        # save the file
        fig.savefig(os.path.join(save, save_subj, "Familiarity_" + subj + ".svg"))
        plt.close(fig)

        df_gew_save = df.groupby(['condition', 'emotion', 'geneva_wheel', 'audio']).count().reset_index().iloc[:,:4]
        df_gew_save['strength'] = df.groupby(['condition', 'emotion', 'audio']).mean().reset_index()['geneva_slider']
        df_gew_save['familiarity'] = df.groupby(['condition', 'audio']).mean().reset_index()['familiarity']
        df_gew_save['dominance'] = df.groupby(['condition', 'audio']).mean().reset_index()['dominance']

        df_gew_save['arousal_mean'] = 'NaN'
        df_gew_save['arousal_std'] = 'NaN'
        df_gew_save['arousal_min'] = 'NaN'
        df_gew_save['arousal_max'] = 'NaN'

        df_gew_save['valence_mean'] = 'NaN'
        df_gew_save['valence_std'] = 'NaN'
        df_gew_save['valence_min'] = 'NaN'
        df_gew_save['valence_max'] = 'NaN'
        df_gew_save['subj'] = subj

        for file_arousal, file_valence in zip(arousal_ratings, valence_ratings):
            df_gew_save.loc[df_gew_save['audio'] == file_arousal[:-16] + '.wav', 'arousal_mean'] = pd.read_csv(os.path.join(source_data, subj, file_arousal), decimal='.', sep=';')['slider_pos'][30:].mean()
            df_gew_save.loc[df_gew_save['audio'] == file_arousal[:-16] + '.wav', 'arousal_std'] = pd.read_csv(os.path.join(source_data, subj, file_arousal), decimal='.', sep=';')['slider_pos'][30:].std()

            df_gew_save.loc[df_gew_save['audio'] == file_arousal[:-16] + '.wav', 'arousal_min'] = pd.read_csv(os.path.join(source_data, subj, file_arousal), decimal='.', sep=';')['slider_pos'][30:].min()
            df_gew_save.loc[df_gew_save['audio'] == file_arousal[:-16] + '.wav', 'arousal_max']= pd.read_csv(os.path.join(source_data, subj, file_arousal), decimal='.', sep=';')['slider_pos'][30:].max()

            df_gew_save.loc[df_gew_save['audio'] == file_valence[:-16] + '.wav', 'valence_mean'] = pd.read_csv(os.path.join(source_data, subj, file_valence), decimal='.', sep=';')['slider_pos'][30:].mean()
            df_gew_save.loc[df_gew_save['audio'] == file_valence[:-16] + '.wav', 'valence_std'] = pd.read_csv(os.path.join(source_data, subj, file_valence), decimal='.', sep=';')['slider_pos'][30:].std()

            df_gew_save.loc[df_gew_save['audio'] == file_valence[:-16] + '.wav', 'valence_min'] = pd.read_csv(os.path.join(source_data, subj, file_valence), decimal='.', sep=';')['slider_pos'][30:].min()
            df_gew_save.loc[df_gew_save['audio'] == file_valence[:-16] + '.wav', 'valence_max'] = pd.read_csv(os.path.join(source_data, subj, file_valence), decimal='.', sep=';')['slider_pos'][30:].max()


        df_gew_save.to_csv(os.path.join(save, "df_features.csv"), header = (not os.path.exists(os.path.join(save, "df_features.csv"))), mode = 'a', decimal = ',', sep = ';')


        fig, axes = plt.subplots(len(df['condition'].unique()),1, figsize=(12,10), sharex=False, sharey=True)
        axes = axes.flatten()
        for id_col, (ax, cond) in enumerate(zip(axes, sorted(list(df['condition'].unique())))):
            df_plot = df.loc[df['condition']==cond]
            ax.bar(df_plot['audio'],df_plot['dominance'], color = lst_color_order[id_col])
            ax.set_title(cond + ' Condition')
            ax.set(xlabel='Audiofiles', ylabel= 'Dominance')
            ax.set_ylim(0, 100)
            ax.set_xticks(df_plot['audio'])
            # since rotation=45, anchor the rotation to fix label/tick alignment
            ax.set_xticklabels(df_plot['list_id'], rotation=45, ha='right', rotation_mode='anchor')


        fig.suptitle('Dominance Ratings Subject ID' + subj)
        fig.legend([Line2D([0], [0], color=dict_color['negative'], lw=3),
              Line2D([0], [0], color=dict_color['neutral'], lw=3),
              Line2D([0], [0], color=dict_color['positive'], lw=3)],
              ['negative', 'neutral','positive'],fontsize='xx-large')

        plt.show()
        # set the spacing between subplots
        plt.tight_layout()

        # save the file
        fig.savefig(os.path.join(save, save_subj, "Dominance_" + subj + ".svg"))
        plt.close(fig)


# =============================================================================
# Plot Grand Average Valence and Arousal
# =============================================================================

fig, axes = plt.subplots(int(np.round(len(dict_valence.keys())/6)), 7, figsize=(25,15), sharex=False, sharey=True)
axes = axes.flatten()
for id_col, (ax, segment) in enumerate(zip(axes, sorted(list(dict_valence.keys())))):
    scale = 'Valence'
    mean = dict_valence[segment].mean(axis = 0)
    std  = dict_valence[segment].std(axis = 0)
    if 'negative' in segment:  
        condition = 'NEGATIVE'
        color = dict_color['negative']
        file_id = segment[26:28]
    elif 'positive' in segment: 
        condition = 'POSITIVE'
        color = dict_color['positive']
        file_id = segment[32:34]
    elif 'neutral' in segment:
        condition = 'NEUTRAL'
        color = dict_color['neutral']
        file_id = segment[25:27]

    ax.plot(np.array(range(0, dict_valence[segment].shape[-1])), mean, color = color)           
    ax.fill_between(np.array(range(0, dict_valence[segment].shape[-1])), mean + std, mean - std, color = color, alpha = 0.2)
    if ax in axes[len(sorted(list(dict_valence.keys())))-7:len(sorted(list(dict_valence.keys())))]:
        ax.set_xlabel('Time in sec', fontsize=fs, fontweight = 'bold')
    if id_col in [0,7,14,21,28, 35]:
        ax.set_ylabel('Scale (1 - 100)', fontsize=fs, fontweight = 'bold')
    ax.set_title(condition[:3] + ' ' + file_id + ' ($M$=' + str(np.round(dict_valence[segment].mean(axis = 0).mean(axis = 0),2)) + '+/-' + str(np.round(dict_valence[segment].std(axis = 0).std(axis = 0),2)) + ')', fontsize=fs, fontweight = 'bold')
    ax.set_ylim(0, 100)
    ax.set_yticks(np.linspace(0,100,6))
    ax.set_yticklabels(np.linspace(0,100,6), fontsize = fs -2, fontweight='bold',c = 'black')
    ax.set_xticks(np.linspace(0,dict_valence[segment].shape[-1],6))
    x_labels = [int(i) for i in list(np.round(np.linspace(0,dict_valence[segment].shape[-1],6),0))]
    ax.set_xticklabels(x_labels, fontsize = fs - 2, fontweight='bold',c = 'black')

for i_ax, ax in enumerate(axes[len(sorted(list(dict_valence.keys()))):]):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False) 
    ax.axis('off')
    if i_ax == 0:
        ax.legend([Line2D([0], [0], color=dict_color['negative'], lw=3),
          Line2D([0], [0], color=dict_color['neutral'], lw=3),
          Line2D([0], [0], color=dict_color['positive'], lw=3)],
          ['negative (NEG)', 'neutral (NEU)','positive (POS)'],fontsize='xx-large',
          loc ='upper left')

# set the spacing between subplots
plt.tight_layout(rect = [0, 0, 1, 0.96],h_pad=2)
fig.suptitle('Grand Average over Participants for ' + scale, fontsize=fs + 10, fontweight = 'bold')
plt.show()
# save the file
fig.savefig(os.path.join(save,"Grand_Average_Valence.svg"))
plt.close(fig)    


fig, axes = plt.subplots(int(np.round(len(dict_arousal.keys())/6)), 7, figsize=(25,15), sharex=False, sharey=True)
axes = axes.flatten()
for id_col, (ax, segment) in enumerate(zip(axes, sorted(list(dict_arousal.keys())))):
    scale = 'Arousal'
    mean = dict_arousal[segment].mean(axis = 0)
    std  = dict_arousal[segment].std(axis = 0)
    if 'negative' in segment:  
        condition = 'NEGATIVE'
        color = dict_color['negative']
        file_id = segment[26:28]
    elif 'positive' in segment: 
        condition = 'POSITIVE'
        color = dict_color['positive']
        file_id = segment[32:34]
    elif 'neutral' in segment:
        condition = 'NEUTRAL'
        color = dict_color['neutral']
        file_id = segment[25:27]

    ax.plot(np.array(range(0, dict_arousal[segment].shape[-1])), mean, color = color)           
    ax.fill_between(np.array(range(0, dict_arousal[segment].shape[-1])), mean + std, mean - std, color = color, alpha = 0.2)
    if ax in axes[len(sorted(list(dict_arousal.keys())))-7:len(sorted(list(dict_arousal.keys())))]:
        ax.set_xlabel('Time in sec', fontsize=fs, fontweight = 'bold')
    if id_col in [0,7,14,21,28, 35]:
        ax.set_ylabel('Scale (1 - 100)', fontsize=fs, fontweight = 'bold')
    # format text
    ax.set_title(condition[:3] + ' ' + file_id + ' ($M$=' + str(np.round(dict_arousal[segment].mean(axis = 0).mean(axis = 0),2)) + '+/-' + str(np.round(dict_arousal[segment].std(axis = 0).std(axis = 0),2)) + ')', fontsize=fs, fontweight = 'bold')
    ax.set_ylim(0, 100)
    ax.set_yticks(np.linspace(0,100,6))
    ax.set_yticklabels(np.linspace(0,100,6), fontsize = fs - 2, fontweight='bold',c = 'black')
    ax.set_xticks(np.linspace(0,dict_arousal[segment].shape[-1],6))
    x_labels = [int(i) for i in list(np.round(np.linspace(0,dict_arousal[segment].shape[-1],6),0))]
    ax.set_xticklabels(x_labels, fontsize = fs - 2, fontweight='bold',c = 'black')
    
for i_ax, ax in enumerate(axes[len(sorted(list(dict_arousal.keys()))):]):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False) 
    ax.axis('off')
    if i_ax == 0:
        ax.legend([Line2D([0], [0], color=dict_color['negative'], lw=3),
          Line2D([0], [0], color=dict_color['neutral'], lw=3),
          Line2D([0], [0], color=dict_color['positive'], lw=3)],
          ['negative (NEG)', 'neutral (NEU)','positive (POS)'],fontsize='xx-large',
          loc ='upper left')
# set the spacing between subplots
plt.tight_layout(rect = [0, 0, 1, 0.96],h_pad=2)
fig.suptitle('Grand Average over Participants for ' + scale, fontsize=fs + 10, fontweight = 'bold')
# save the file
plt.show()
fig.savefig(os.path.join(save, "Grand_Average_Arousal.svg"))
plt.close(fig)    

if not os.path.exists(os.path.join(save,'dict_valence.pickle')):
    with open(os.path.join(save,'dict_valence.pickle'), 'wb') as file:
        pickle.dump(dict_valence, file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save,'dict_arousal.pickle'), 'wb') as file:
        pickle.dump(dict_arousal, file, protocol=pickle.HIGHEST_PROTOCOL)
