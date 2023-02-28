
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 22:10:07 2021

@author: hirning
"""
# =============================================================================
# Import Packages
# =============================================================================
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import paired_distances, euclidean_distances
import seaborn as sns
import pickle
from scipy.spatial import distance as scdist
import config_analysis

# =============================================================================
# Load data 
# =============================================================================
bids_root = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
data = os.path.join(bids_root,'derivatives', 'ratings')
save = os.path.join(bids_root,'derivatives', 'heatmaps')
if not os.path.exists(save):
    os.makedirs(save)
# =============================================================================
# 3D Scatterplot Interactive
# =============================================================================
#normalized
#df = pd.read_csv(os.path.join(data,'preprocessed_features.csv'), decimal=',', sep=';')
#non-normalized
df = pd.read_csv(os.path.join(data,'df_features.csv'), decimal=',', sep=';')
 
condition_list = df.loc[:, 'condition']
audio_list = df.loc[:, 'audio']
subj_list = df.loc[:, 'subj']
drop = [col for col in df.columns if col not in ['valence_mean', 'arousal_mean', 'dominance']]

df = df.drop(drop, axis = 1)
df = df.apply(pd.to_numeric)

cols = df.columns
titles = ['Dominance', 'Arousal', 'Valence']
metric = 'cosine'

df['condition'] = condition_list
df['audio'] = audio_list 
df['subj'] = subj_list
# =============================================================================
# Plotting Heatmap Distance
# =============================================================================
columns_heat = sorted(list(df.audio.unique()))
columns_heat = ['positive_' + row[-6:-4] if 'positive' in row else row[17:-4] for row in columns_heat]
columns_heat = [ row[:-3] + ' '+ str(int(row[-2:]) + 1) for row in columns_heat]
df.audio = df.audio.replace(dict(zip(sorted(list(df.audio.unique())),columns_heat)))
df_heat_sorted =  pd.DataFrame(columns = columns_heat, index = columns_heat) 

# =============================================================================
# Combined V - A - D
# =============================================================================
for start_audio in sorted(list(df.audio.unique())):
    for ref_audio in sorted(list(df.audio.unique())):
        #print(start_audio, ref_audio)
        #distance = dist.pairwise(np.array(df.loc[:,cols])[list(np.where(df.audio == start_audio)[0])],np.array(df.loc[:,cols])[list(np.where(df.audio == ref_audio)[0])])
        #np.array(df.loc[:,'subj'])[list(np.where(df.audio == start_audio)[0])]
        #np.array(df.loc[:,'subj'])[list(np.where(df.audio == ref_audio)[0])]
        intersection = set(np.array(df.loc[:,'subj'])[list(np.where(df.audio == start_audio)[0])]).intersection(np.array(df.loc[:,'subj'])[list(np.where(df.audio == ref_audio)[0])])
        #print('LEN INTERSECTION', len(intersection))
        distance = scdist.cdist(np.array(df.loc[:,cols])[list(np.where((df.audio == start_audio) & (df.subj.isin(intersection)))[0])],np.array(df.loc[:,cols])[list(np.where((df.audio == ref_audio) & (df.subj.isin(intersection)))[0])], metric)
        #print(distance.mean())
        df_heat_sorted.loc[df_heat_sorted.index == ref_audio, start_audio] = distance.mean()

df_heat_sorted = df_heat_sorted.apply(pd.to_numeric)
for half in [True, False]:
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Combined Valence - Arousal - Dominance', fontsize = 25, fontweight = 'bold', fontname="Times New Roman")    
    if half: 
        matrix = np.triu(df_heat_sorted) # take upper correlation matrix
        plot = sns.heatmap(df_heat_sorted, mask=matrix)
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=15)
        cax.set_ylabel('Cosine Distance', size = 18, fontname="Times New Roman", fontweight = 'bold')
    else:
        plot = sns.heatmap(df_heat_sorted)
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=15)
        cax.set_ylabel('Cosine Distance', size = 18, fontname="Times New Roman", fontweight = 'bold')
    plot.set_yticklabels(columns_heat, size = 15,fontname="Times New Roman")
    plot.set_xticklabels(columns_heat, size = 15,fontname="Times New Roman")
    fig.tight_layout()
    plt.savefig(os.path.join(save, 'Dis_'+ metric + '_combined_VAD_half_'+ str(half) +'.svg'))
    plt.close() 

# =============================================================================
# Valence
# =============================================================================

df_heat_valence =  pd.DataFrame(columns = columns_heat, index = columns_heat) 

for start_audio in sorted(list(df.audio.unique())):
    for ref_audio in sorted(list(df.audio.unique())):
        #print(start_audio, ref_audio)
        
        intersection = set(np.array(df.loc[:,'subj'])[list(np.where(df.audio == start_audio)[0])]).intersection(np.array(df.loc[:,'subj'])[list(np.where(df.audio == ref_audio)[0])])
        print('LEN INTERSECTION', len(intersection))
        #print(np.array(df.loc[:,'subj'])[list(np.where((df.audio == start_audio) & (df.subj.isin(intersection)))[0])])
        #print(np.array(df.loc[:,'subj'])[list(np.where((df.audio == ref_audio) & (df.subj.isin(intersection)))[0])])
        if (start_audio == 'neutral 10') and (ref_audio == 'positive 1'):
            with open(os.path.join(save,'ratings_neutral_10.pickle'), 'wb') as file:
                pickle.dump(np.array(df.loc[:,'valence_mean'])[list(np.where((df.audio == start_audio) & (df.subj.isin(intersection)))[0])], file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(save,'ratings_positive_01.pickle'), 'wb') as file:
                pickle.dump(np.array(df.loc[:,'valence_mean'])[list(np.where((df.audio == ref_audio) & (df.subj.isin(intersection)))[0])], file, protocol=pickle.HIGHEST_PROTOCOL)

        distance = scdist.cosine(np.array(df.loc[:,'valence_mean'])[list(np.where((df.audio == start_audio) & (df.subj.isin(intersection)))[0])],np.array(df.loc[:,'valence_mean'])[list(np.where((df.audio == ref_audio) & (df.subj.isin(intersection)))[0])])
        metric = 'cosine'
        #print(distance.mean())
        df_heat_valence.loc[df_heat_valence.index == ref_audio, start_audio] = distance
df_heat_valence = df_heat_valence.apply(pd.to_numeric)
for half in [True, False]:
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Valence', fontsize = 25, fontweight = 'bold', fontname="Times New Roman")
    if half: 
        matrix = np.triu(df_heat_valence) # take upper correlation matrix
        plot = sns.heatmap(df_heat_valence, mask=matrix, cbar_kws={'shrink': 1.0, 'label': 'Cosine Distance'})
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=15)
        cax.set_ylabel('Cosine Distance', size = 15, fontname="Times New Roman", fontweight = 'bold')
    else:
        plot = sns.heatmap(df_heat_valence, cbar_kws={'shrink': 1.0, 'label': 'Cosine Distance'})
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=15)
        cax.set_ylabel('Cosine Distance', size = 18, fontname="Times New Roman", fontweight = 'bold')
    #plot = sns.heatmap(df_heat_valence, cmap="magma_r")
    plot.set_yticklabels(columns_heat, size = 15,fontname="Times New Roman")
    plot.set_xticklabels(columns_heat, size = 15,fontname="Times New Roman")
    fig.tight_layout()
    plt.savefig(os.path.join(save, 'Dis_'+ metric + '_valence_half_'+ str(half) +'.svg'))
    plt.close() 


# =============================================================================
# Arousal
# =============================================================================

df_heat_arousal =  pd.DataFrame(columns = columns_heat, index = columns_heat) 

for start_audio in sorted(list(df.audio.unique())):
    for ref_audio in sorted(list(df.audio.unique())):
        #print(start_audio, ref_audio)
        intersection = set(np.array(df.loc[:,'subj'])[list(np.where(df.audio == start_audio)[0])]).intersection(np.array(df.loc[:,'subj'])[list(np.where(df.audio == ref_audio)[0])])
        print('LEN INTERSECTION', len(intersection))
        distance = scdist.cosine(np.array(df.loc[:,'arousal_mean'])[list(np.where((df.audio == start_audio) & (df.subj.isin(intersection)))[0])],np.array(df.loc[:,'arousal_mean'])[list(np.where((df.audio == ref_audio) & (df.subj.isin(intersection)))[0])])
        metric = 'cosine' #sqeuclidean
        #print(distance.mean())
        df_heat_arousal.loc[df_heat_arousal.index == ref_audio, start_audio] = distance
df_heat_arousal = df_heat_arousal.apply(pd.to_numeric)
for half in [True, False]:
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Arousal', fontsize = 25, fontweight = 'bold', fontname="Times New Roman")
    if half: 
        matrix = np.triu(df_heat_arousal) # take upper correlation matrix
        plot = sns.heatmap(df_heat_arousal, mask=matrix)
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=15)
        cax.set_ylabel('Cosine Distance', size = 15, fontname="Times New Roman", fontweight = 'bold')
    else:
        plot = sns.heatmap(df_heat_arousal)
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=15)
        cax.set_ylabel('Cosine Distance', size = 18, fontname="Times New Roman", fontweight = 'bold')
    plot.set_yticklabels(columns_heat, size = 15,fontname="Times New Roman")
    plot.set_xticklabels(columns_heat, size = 15,fontname="Times New Roman")
    fig.tight_layout()
    plt.savefig(os.path.join(save, 'Dis_'+ metric + '_arousal_half_'+ str(half) +'.svg'))
    plt.close() 


# =============================================================================
# Dominance
# =============================================================================

df_heat_dominance =  pd.DataFrame(columns = columns_heat, index = columns_heat) 

for start_audio in sorted(list(df.audio.unique())):
    for ref_audio in sorted(list(df.audio.unique())):
        #print(start_audio, ref_audio)
        intersection = set(np.array(df.loc[:,'subj'])[list(np.where(df.audio == start_audio)[0])]).intersection(np.array(df.loc[:,'subj'])[list(np.where(df.audio == ref_audio)[0])])
        print('LEN INTERSECTION', len(intersection))
        distance = scdist.cosine(np.array(df.loc[:,'dominance'])[list(np.where((df.audio == start_audio) & (df.subj.isin(intersection)))[0])],np.array(df.loc[:,'dominance'])[list(np.where((df.audio == ref_audio) & (df.subj.isin(intersection)))[0])])
        metric = 'cosine'
        #print(distance.mean())
        df_heat_dominance.loc[df_heat_dominance.index == ref_audio, start_audio] = distance
df_heat_dominance = df_heat_dominance.apply(pd.to_numeric)
for half in [True, False]:
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Dominance', fontsize = 25, fontweight = 'bold', fontname="Times New Roman")
    if half: 
        matrix = np.triu(df_heat_dominance) # take upper correlation matrix
        plot = sns.heatmap(df_heat_dominance, mask=matrix)
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=15)
        cax.set_ylabel('Cosine Distance', size = 18, fontname="Times New Roman", fontweight = 'bold')
    else:
        plot = sns.heatmap(df_heat_dominance)
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=15)
        cax.set_ylabel('Cosine Distance', size = 18, fontname="Times New Roman", fontweight = 'bold')
    plot.set_yticklabels(columns_heat, size = 15,fontname="Times New Roman")
    plot.set_xticklabels(columns_heat, size = 15,fontname="Times New Roman")
    fig.tight_layout()
    plt.savefig(os.path.join(save, 'Dis_'+ metric + '_dominance_half_'+ str(half) +'.svg'))
    plt.close() 
