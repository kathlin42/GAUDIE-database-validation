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
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import config_analysis
# =============================================================================
# Load data 
# =============================================================================
p = 0.05
n_comp = 18 
corr_p = p/n_comp
bids_root = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
data = os.path.join(bids_root, 'derivatives', 'ratings')
df = pd.read_csv(os.path.join(data, 'df_features.csv'), decimal='.', sep=';')
df_corr = pd.DataFrame(columns = ['condition', 'pairs', 'stats', 'p'])
p_vals = []



for con in df.condition.unique():

    a = df.loc[df.condition == con, 'arousal_mean']
    v = df.loc[df.condition == con, 'valence_mean']
    f = df.loc[df.condition == con, 'familiarity']
    d = df.loc[df.condition == con, 'dominance']
    #print(con, 'A - V', stats.pearsonr(a, v))
    print(con, 'A - F', stats.pearsonr(a, f))
    #print(con, 'A - D', stats.pearsonr(a, d))
    print(con, 'V - F', stats.pearsonr(v, f))
    #print(con, 'V - D', stats.pearsonr(v, d))
    print(con, 'D - F', stats.pearsonr(d, f))
    #comparision = ['A-V', 'A-F', 'A-D', 'V-F', 'V-D', 'D-F']
    comparision = ['A-F', 'V-F', 'D-F']
    fig = plt.figure(figsize=(15, 10), facecolor='white')   
    fig.suptitle('Correlations between Scales - ' + con, fontsize = 20, fontweight='bold')
    #for idx, (var1, var2) in enumerate([(a,v),(a,f),(a,d),(v,f),(v,d),(d,f)]):
    for idx, (var1, var2) in enumerate([(a,f),(v,f),(d,f)]):
        p_vals.append(stats.pearsonr(var1, var2)[1])        
        # plot result
        ax = fig.add_subplot(3,1, idx + 1)
        ax.scatter(var1, var2 , alpha=.5)
        ax.scatter(np.mean(var1), np.mean(var2), marker="+", s=100, c='black', label='mean')
        if comparision[idx] == 'A-V':
            xlabel = 'Arousal'
            ylabel = 'Valence'
        elif comparision[idx] == 'A-F':
            xlabel = 'Arousal'
            ylabel = 'Familiarity'
        elif comparision[idx] == 'A-D':
            xlabel = 'Arousal'
            ylabel = 'Dominance'
            
        elif comparision[idx] == 'V-F':
            xlabel = 'Valence'
            ylabel = 'Familiarity'
        elif comparision[idx] == 'V-D':
            xlabel = 'Valence'
            ylabel = 'Dominance'
        elif comparision[idx] == 'D-F':
            xlabel = 'Dominance'
            ylabel = 'Familiarity'
            
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel,fontsize=14, fontweight='bold')
        ax.plot(np.unique(var1), np.poly1d(np.polyfit(var1, var2, 1))(np.unique(var1)))
        #sns.lmplot(x=comparision[idx][0], y=comparision[idx][-1], data=pd.DataFrame({comparision[idx][0]: var1, comparision[idx][-1]: var2}), ax = ax)
    plt.tight_layout()
    plt.savefig(os.path.join(data,'correlation_'+ con +'.svg'))
    plt.show()
    plt.close()

    temp = {'condition': [con] * 3,
    	'pairs': comparision,
    	'stats': [np.round(stats.pearsonr(a, f)[0], 3), np.round(stats.pearsonr(v, f)[0],3), np.round(stats.pearsonr(d, f)[0],3)],
    	'p': [np.round(stats.pearsonr(a, f)[1], 3), np.round(stats.pearsonr(v, f)[1],3), np.round(stats.pearsonr(d, f)[1],3)]}
    

       
    #temp = {'condition': [con] * 6,
    	#'pairs': comparision,
    	#'stats': [np.round(stats.pearsonr(a, v)[0], 3), np.round(stats.pearsonr(a, f)[0], 3), np.round(stats.pearsonr(a, d)[0],3),
    #              np.round(stats.pearsonr(v, f)[0],3), np.round(stats.pearsonr(v, d)[0], 3), np.round(stats.pearsonr(d, f)[0],3)],
    	#'p': [np.round(stats.pearsonr(a, v)[1], 3), np.round(stats.pearsonr(a, f)[1], 3), np.round(stats.pearsonr(a, d)[1],3),
    #              np.round(stats.pearsonr(v, f)[1],3), np.round(stats.pearsonr(v, d)[1], 3), np.round(stats.pearsonr(d, f)[1],3)]}
    
    df_corr = df_corr.append(pd.DataFrame(temp), ignore_index=True)

corrected_pvals = multipletests(p_vals, alpha = 0.05, method = 'fdr_bh', is_sorted = False) 
df_corr['Stats_Sig'] = corrected_pvals[0]
df_corr['Corr_P'] = np.round(corrected_pvals[1], 3)
df_corr.to_csv(os.path.join(data, 'corr_results.csv'), sep=';', decimal=',', header=True)
         