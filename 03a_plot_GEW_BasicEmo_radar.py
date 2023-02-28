# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:03:22 2022

@author: hirning
"""

import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import config_analysis
bids_root = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
data = os.path.join(bids_root, 'derivatives', 'ratings')

mpl.rc('font',family='Times New Roman', weight = 'bold')
lst_color = ['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet']
dict_color = {'positive': lst_color[2],
              'neutral' : lst_color[0],
              'negative' : lst_color[1]}
fs = 14

df = pd.read_csv(os.path.join(data, 'df_features.csv'), decimal=',', sep=';')

list_emo = ['Angst', 'Ekel', 'Trauer', 'Wut', '�berraschung', 'Freude']
list_gew = list(df['geneva_wheel'].unique())
df = pd.get_dummies(df, columns = ['emotion', 'geneva_wheel'])
for i, strength_response in enumerate(list_gew):
    df.loc[i, 'geneva_wheel_' + strength_response] = df.loc[i, 'geneva_wheel_' + strength_response] *  df.loc[i, 'strength']
lst_color=['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet']

labels = ['Fear', 'Disgust', 'Joy', 'Sadness', 'Anger', 'Surprise']
labels_gew = ['Fear', 'Amusement', 'Regret', 'Disgust', 'Disappointment', 'Relief', 'Joy', 'Hatred', 'Interest', 'Compassion', 'Shame', 'Guilt',
              'Pride', 'Sadness', 'Contempt', 'Pleasure', 'Anger', 'Contentment']

emo_col = [col for col in df.columns if not 'geneva' in col]
gew_col = [col for col in df.columns if not 'emotion' in col]
df_emo = df.loc[:, emo_col] 
df_gew = df.loc[:, gew_col] 

df_gew.rename(columns = dict(zip(list(df_gew.columns[-len(labels_gew):]), labels_gew)), inplace = True)
df_emo.rename(columns = dict(zip(list(df_emo.columns[-len(labels):]), labels)), inplace = True)

labels = ['Joy', 'Surprise','Fear',  'Anger',  'Sadness','Disgust']
labels_gew = ['Interest', 'Amusement', 'Pride', 'Joy', 'Pleasure','Contentment', 'Relief', 'Compassion', 'Sadness', 'Guilt', 'Regret','Shame', 'Disappointment','Fear',  'Disgust',  'Contempt', 'Hatred' , 'Anger']
df_group = df_gew.groupby(['audio', 'condition']).mean().reset_index()
df_group['audio'] = ['positive ' + str(int(row[-6:-4]) + 1) if 'positive' in row else row[17:-7] + ' ' + str(int(row[-6:-4]) +1) for row in df_group['audio']]

fig, axes = plt.subplots(1,3,figsize=(20, 20), subplot_kw=dict(polar=True))

for i_cond, cond in enumerate(['negative', 'neutral', 'positive']):
    ax = axes.flatten()[i_cond]
    for audio in df_group['audio'].unique():
        if cond in audio:
            if 'negative' in audio:  
                condition = 'NEGATIVE'
                color = dict_color['negative']
            elif 'positive' in audio: 
                condition = 'POSITIVE'
                color = dict_color['positive']
            elif 'neutral' in audio:
                condition = 'NEUTRAL'
                color =  dict_color['neutral']
            values = df_group.loc[df_group['audio'] == audio, labels_gew].values[0].tolist()
            
            # Number of variables we're plotting.
            num_vars = len(labels_gew)    
            # Split the circle into even parts and save the angles
            # so we know where to put each axis.
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            # Draw the outline of our data.
            ax.plot(angles, values, color=color, linewidth=1)

            # Fill it in.
            ax.fill(angles, values, color=color, alpha=0.25)
        # Fix axis to go in the right order and start at 12 o'clock.
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label.
        ax.set_thetagrids(np.degrees(angles), labels_gew)
        
        # Go through labels and adjust alignment based on where
        # it is in the circle.
        for label, angle in zip(ax.get_xticklabels(), angles):
          if angle in (0, np.pi):
            label.set_horizontalalignment('center')
          elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
          else:
            label.set_horizontalalignment('right')
        
        # Ensure radar goes from 0 to 100.
        ax.set_ylim(0, 1)
        # You can also set gridlines manually like this:
        # ax.set_rgrids([20, 40, 60, 80, 100])
        
        # Set position of y-labels (0-100) to be in the middle
        # of the first two axes.
        ax.set_rlabel_position(180 / num_vars)
        for r_label in ax.get_yticklabels():
            r_label.set(fontsize = 16, fontweight='bold',c = 'black')
            
        for r_label in ax.get_xticklabels():
            r_label.set(fontsize = 16, fontweight='bold',c = 'black')
        
        ax.set_title(condition, y=1.10, fontsize = 20, fontweight='bold',c = 'black')   
        
#fig.suptitle('Distribution of the Geneva Wheel Ratings per Condition', fontsize = 18, fontweight='bold',c = 'black')
fig.tight_layout()
plt.show()
fig.savefig(os.path.join(data, 'GEW_ratings.svg'))
plt.close(fig)

# =============================================================================
# Basic Emotion 
# =============================================================================

df_group = df_emo.groupby(['audio', 'condition']).mean().reset_index()
df_group['audio'] = ['positive ' + str(int(row[-6:-4]) + 1) if 'positive' in row else row[17:-7] + ' ' + str(int(row[-6:-4]) +1) for row in df_group['audio']]
fig, axes = plt.subplots(1,3,figsize=(20, 20), subplot_kw=dict(polar=True))

for i_cond, cond in enumerate(['negative', 'neutral', 'positive']):
    ax = axes.flatten()[i_cond]
    for audio in df_group['audio'].unique():
        if cond in audio:
            if 'negative' in audio:  
                condition = 'NEGATIVE'
                color = dict_color['negative']
            elif 'positive' in audio:
                condition = 'POSITIVE'
                color = dict_color['positive']
            elif 'neutral' in audio:
                condition = 'NEUTRAL'
                color =  dict_color['neutral']
            values = df_group.loc[df_group['audio'] == audio, labels].values[0].tolist()
            
            # Number of variables we're plotting.
            num_vars = len(labels)    
            # Split the circle into even parts and save the angles
            # so we know where to put each axis.
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            # Draw the outline of our data.
            ax.plot(angles, values, color=color, linewidth=1)
            # Fill it in.
            ax.fill(angles, values, color=color, alpha=0.25)
        # Fix axis to go in the right order and start at 12 o'clock.
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label.
        ax.set_thetagrids(np.degrees(angles), labels)
        
        # Go through labels and adjust alignment based on where
        # it is in the circle.
        for label, angle in zip(ax.get_xticklabels(), angles):
          if angle in (0, np.pi):
            label.set_horizontalalignment('center')
          elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
          else:
            label.set_horizontalalignment('right')
        
        # Ensure radar goes from 0 to 100.
        ax.set_ylim(0, 1)
        # You can also set gridlines manually like this:
        # ax.set_rgrids([20, 40, 60, 80, 100])
        
        # Set position of y-labels (0-100) to be in the middle
        # of the first two axes.
        ax.set_rlabel_position(180 / num_vars)
        for r_label in ax.get_yticklabels():
            r_label.set(fontsize = 16, fontweight='bold',c = 'black')
            
        for r_label in ax.get_xticklabels():
            r_label.set(fontsize = 16, fontweight='bold',c = 'black')
        ax.set_title(condition, y=1.08, fontsize = 20, fontweight='bold',c = 'black')   


#fig.suptitle('Distribution of the Basic Emotions Ratings per Condition', fontsize = 18, fontweight='bold',c = 'black')
fig.tight_layout(h_pad=2)
plt.show()
fig.savefig(os.path.join(data , 'Emo_ratings.svg'))
plt.close(fig)


# =============================================================================
# Per Audio Sequence
# =============================================================================

df_group = df_gew.groupby(['audio', 'condition']).mean().reset_index()
df_group['audio'] = ['positive ' + str(int(row[-6:-4]) + 1) if 'positive' in row else row[17:-7] + ' ' + str(int(row[-6:-4]) +1) for row in df_group['audio']]

for i_cond, cond in enumerate(['negative', 'neutral', 'positive']):
    df_plot = df_group.loc[df_group['condition'] == cond]
    if cond in 'neutral':
        fig, axes = plt.subplots(int(np.ceil(len(df_plot.audio.unique())/4)),4,figsize=(30, 28), subplot_kw=dict(polar=True))
    else:
        fig, axes = plt.subplots(int(np.ceil(len(df_plot.audio.unique())/4)),4,figsize=(30, 20), subplot_kw=dict(polar=True))
    for i_audio, audio in enumerate(df_plot['audio'].unique()):
        ax = axes.flatten()[i_audio]
        if cond in audio:
            if 'negative' in audio:  
                condition = 'NEGATIVE'
                color = dict_color['negative']
                pad = -2
            elif 'positive' in audio: 
                condition = 'POSITIVE'
                color = dict_color['positive']
                pad = -2
            elif 'neutral' in audio:
                condition = 'NEUTRAL'
                color = dict_color['neutral']
                pad = -3
            values = df_plot.loc[df_plot['audio'] == audio, labels_gew].values[0].tolist()
            
            # Number of variables we're plotting.
            num_vars = len(labels_gew)    
            # Split the circle into even parts and save the angles
            # so we know where to put each axis.
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            # Draw the outline of our data.
            ax.plot(angles, values, color=color, linewidth=1)

            # Fill it in.
            ax.fill(angles, values, color=color, alpha=0.25)
        # Fix axis to go in the right order and start at 12 o'clock.
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label.
        ax.set_thetagrids(np.degrees(angles), labels_gew)

        # Ensure radar goes from 0 to 100.
        ax.set_ylim(0, 1)
        
        # You can also set gridlines manually like this:
        # ax.set_rgrids([20, 40, 60, 80, 100])
        
        # Set position of y-labels (0-100) to be in the middle
        # of the first two axes.
        ax.set_rlabel_position(180 / num_vars)
        for r_label in ax.get_yticklabels():
            r_label.set(fontsize = 18, fontweight='bold',c = 'black')
            
        for r_label in ax.get_xticklabels():
            r_label.set(fontsize = 21, fontweight='bold',c = 'black')
        # Go through labels and adjust alignment based on where
        # it is in the circle.
        for label, angle in zip(ax.get_xticklabels(), angles):
          if angle in (0, np.pi):
            label.set_horizontalalignment('center')
          elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
          else:
            label.set_horizontalalignment('right')
            
        ax.set_title(audio.replace('_', ' '), y=1.08, fontsize = 28, fontweight='bold',c = 'black')   
        
    for i_ax, ax in enumerate(axes.flatten()[len(df_plot['audio'].unique()):]):
        ax.set_visible(False)
    #fig.suptitle('Distribution of the Geneva Wheel Ratings for the ' + condition + ' Condition', fontsize = 18, fontweight='bold',c = 'black')
    fig.tight_layout(pad=pad)
    plt.show()
    fig.savefig(os.path.join(data, 'GEW_ratings_'+ condition +'.svg'), bbox_inches = 'tight')
    plt.close(fig)
# =============================================================================
# Basic Emotion 
# =============================================================================

df_group = df_emo.groupby(['audio', 'condition']).mean().reset_index()
df_group['audio'] = ['positive ' + str(int(row[-6:-4]) + 1) if 'positive' in row else row[17:-7] + ' ' + str(int(row[-6:-4]) +1) for row in df_group['audio']]

for i_cond, cond in enumerate(['negative', 'neutral', 'positive']):
    df_plot = df_group.loc[df_group['condition'] == cond]
    if cond in 'neutral':
        fig, axes = plt.subplots(int(np.ceil(len(df_plot.audio.unique())/4)),4,figsize=(25, 25), subplot_kw=dict(polar=True))
    else:
        fig, axes = plt.subplots(int(np.ceil(len(df_plot.audio.unique())/4)),4,figsize=(25, 18), subplot_kw=dict(polar=True))
    for i_audio, audio in enumerate(df_plot['audio'].unique()):
        ax = axes.flatten()[i_audio]
        if cond in audio:
            if 'negative' in audio:  
                condition = 'NEGATIVE'
                color = dict_color['negative']
              
            elif 'positive' in audio: 
                condition = 'POSITIVE'
                color = dict_color['positive']
          
            elif 'neutral' in audio:
                condition = 'NEUTRAL'
                color = dict_color['neutral']
                
            values = df_plot.loc[df_plot['audio'] == audio, labels].values[0].tolist()
            
            # Number of variables we're plotting.
            num_vars = len(labels)    
            # Split the circle into even parts and save the angles
            # so we know where to put each axis.
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            # Draw the outline of our data.
            ax.plot(angles, values, color=color, linewidth=1)

            # Fill it in.
            ax.fill(angles, values, color=color, alpha=0.25)
        # Fix axis to go in the right order and start at 12 o'clock.
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label.
        ax.set_thetagrids(np.degrees(angles), labels)
        
        # Go through labels and adjust alignment based on where
        # it is in the circle.
        for label, angle in zip(ax.get_xticklabels(), angles):
          if angle in (0, np.pi):
            label.set_horizontalalignment('center')
          elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
          else:
            label.set_horizontalalignment('right')
        
        # Ensure radar goes from 0 to 100.
        ax.set_ylim(0, 1)
        # You can also set gridlines manually like this:
        # ax.set_rgrids([20, 40, 60, 80, 100])
        
        # Set position of y-labels (0-100) to be in the middle
        # of the first two axes.
        ax.set_rlabel_position(180 / num_vars)
        for r_label in ax.get_yticklabels():
            r_label.set(fontsize = 22, fontweight='bold',c = 'black')
            
        for r_label in ax.get_xticklabels():
            r_label.set(fontsize = 22, fontweight='bold',c = 'black')
        ax.set_title(audio.replace('_', ' '), y=1.08, fontsize = 28, fontweight='bold',c = 'black')   
        
        
    for i_ax, ax in enumerate(axes.flatten()[len(df_plot['audio'].unique()):]):
        ax.set_visible(False)
        
    #fig.suptitle('Distribution of the Basic Emotions Ratings for the ' + condition + ' Condition', fontsize = 18, fontweight='bold',c = 'black')
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(data, 'Emo_ratings_'+ condition +'.svg'),bbox_inches = 'tight')
    plt.close(fig)