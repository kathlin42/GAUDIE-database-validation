# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 18:02:57 2022

@author: hirning
"""

# =============================================================================
# Packages
# =============================================================================
import pickle
import os
import os
import pandas as pd 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
mpl.rc('font',family='Times New Roman', weight = 'bold')
import pickle
import fontstyle
import helper_plotting as pb
import config_analysis
# =============================================================================
# Data path 
# =============================================================================
bids_root = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
data_path = os.path.join(bids_root, 'derivatives', 'ratings')

# =============================================================================
# Open Pickle
# =============================================================================

with open(os.path.join(data_path,'dict_arousal.pickle'),'rb') as f:
    dict_arousal = pickle.load(f)
with open(os.path.join(data_path,'dict_valence.pickle'),'rb') as f:
    dict_valence = pickle.load(f)
    
scales_names = ['Arousal', 'Valence']
lst_color = ['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet']
dict_color = {'positive': lst_color[2],
              'neutral' : lst_color[0],
              'negative' : lst_color[1]}
for i_scale, dict_rating in enumerate([dict_arousal, dict_valence]):
    # =============================================================================
    # Plot Grand Average Valence and Arousal
    # =============================================================================
    
    fig, axes = plt.subplots(int(np.round(len(dict_rating.keys())/6)), 7, figsize=(25,15), sharex=False, sharey=True)
    axes = axes.flatten()
    
    for id_col, (ax, segment) in enumerate(zip(axes, sorted(list(dict_rating.keys())))):
        scale = scales_names[i_scale]
               
        for sample in range(0,dict_rating[segment].shape[-1]):
            print(sample, 'from', dict_rating[segment].shape[-1])
            dict_boot = pb.bootstrapping(dict_rating[segment][:,sample],
                                         numb_iterations = 5000,
                                         alpha =0.95,
                                         plot_hist =False,
                                         as_dict = True,
                                         func = 'mean')
            if sample == 0: 
                mean = np.array(dict_boot['mean'])
                upper = np.array(dict_boot['upper'])
                lower = np.array(dict_boot['lower'])
            else: 
                mean = np.append(mean, dict_boot['mean'])
                upper = np.append(upper, dict_boot['upper'])
                lower = np.append(lower, dict_boot['lower'])
                 
        if 'negative' in segment:  
            condition = 'NEG'
            color = dict_color['negative']
            file_id = str(int(segment[26:28]) + 1) 
        elif 'positive' in segment: 
            condition = 'POS'
            color = dict_color['positive']
            file_id = str(int(segment[32:34]) + 1)
        elif 'neutral' in segment:
            condition = 'NEU'
            color = dict_color['neutral']
            file_id = str(int(segment[25:27]) + 1)
    
        ax.plot(np.array(range(0, dict_rating[segment].shape[-1])), mean, color = color)           
        ax.fill_between(np.array(range(0, dict_rating[segment].shape[-1])), upper, lower, color = color, alpha = 0.2)

        if ax in axes[len(sorted(list(dict_rating.keys())))-7:len(sorted(list(dict_rating.keys())))]:
            ax.set_xlabel('Time in sec', fontsize=20, fontweight = 'bold',fontname="Times New Roman")
        if id_col in [0,7,14,21,28, 35]:
            ax.set_ylabel('Scale (1 - 100)', fontsize=22, fontweight = 'bold',fontname="Times New Roman")
       
        ax.set_title(condition + ' ' + file_id + ' $M$=' + str(np.round(np.mean(mean),2)) + ' [' + str(np.round(np.mean(lower),2)) + ',' + str(np.round(np.mean(upper),2)) + ']', fontsize=18, fontweight = 'bold',fontname="Times New Roman")  
        ax.set_ylim(0, 100)
        ax.set_yticks(np.linspace(0,100,6))
        ax.set_yticklabels(np.linspace(0,100,6), fontsize = 22, fontweight='bold',c = 'black',fontname="Times New Roman")
        ax.set_xticks(np.linspace(0,dict_rating[segment].shape[-1],6))
        x_labels = [int(i) for i in list(np.round(np.linspace(0,dict_rating[segment].shape[-1],6),0))]
        ax.set_xticklabels(x_labels, fontsize = 22, fontweight='bold',c = 'black',fontname="Times New Roman")
        ax.axvline(x = 30,c = 'grey', linestyle = '--' )
    for i_ax, ax in enumerate(axes[len(sorted(list(dict_rating.keys()))):]):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False) 
        ax.axis('off')
        if i_ax == 0:

            ax.legend([Line2D([0], [0], color=dict_color['negative'], lw=3, ls = 'dashed'),
          Line2D([0], [0], color=dict_color['neutral'], lw=3, ls = 'dashed'),
          Line2D([0], [0], color=dict_color['positive'], lw=3, ls = 'dashed')],
          ['negative (NEG)', 'neutral (NEU)','positive (POS)'],fontsize='xx-large',
              loc ='upper left')
    
    # set the spacing between subplots
    plt.tight_layout(rect = [0, 0, 1, 0.955],h_pad=2)
    fig.suptitle('Grand Average over Participants for ' + scale, fontsize=30, fontweight = 'bold',fontname="Times New Roman")            
    plt.show()
    # save the file
    fig.savefig(os.path.join(data_path, "Grand_Average_" + scale +".svg"))
    plt.close(fig)    
    
                