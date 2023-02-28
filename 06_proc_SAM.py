# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:59:17 2021

@author: hirning
"""
import numpy as np
import pandas as pd
import os
import pickle
from math import sqrt
from matplotlib.lines import Line2D
import helper_plotting as pb
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import pingouin as pg
import config_analysis
mpl.rc('font',family='Times New Roman', weight = 'bold')

ml_feature_selection = False
cluster_permutation = False
            
n_simulation = 5000
fwr_correction = False
boot_size = 5000
boot = 'mean'
lst_color = ['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet']
dict_color = {'positive': lst_color[2],
              'neutral' : lst_color[0],
              'negative' : lst_color[1]}
fs = 14

bids_root = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
source_data = os.path.join(bids_root, 'source_data', 'behavioral')
data = os.path.join(bids_root, 'derivatives', 'ratings')
save_path = os.path.join(bids_root, 'derivatives', 'SAM')

variables = ['valence_mean', 'arousal_mean', 'dominance']
titles = ['Valence', 'Arousal', 'Dominance']
dict_critical = {'valence_mean':['neutral_02', 'neutral_04', 'neutral_07', 'neutral_09', 'positive_07', 'positive_06'],
                 'arousal_mean':['neutral_04','positive_07'], #'neutral_02', 'neutral_04', 'neutral_07', 'neutral_09', 'positive_07', 'positive_06'
                 'dominance':['neutral_05', 'neutral_09',  'positive_08']} 

fig = plt.figure(figsize=(20, 10), facecolor='white')   
fig.suptitle('Emotional Distinctiveness', fontsize = 30, fontweight='bold',fontname="Times New Roman")      

for i_var, setting in enumerate(['valence_based', 'arousal_based', 'dominance_based']):
    
    
    df = pd.read_csv(os.path.join(data, 'df_features.csv'), decimal=',', sep=';')
    condition_list = df.loc[:, ['condition']]
    cols = ['condition','audio','subj'] + [variables[i_var]]
    df = df.loc[:, cols]
    df['audio'] = ['positive_' + row[-6:-4] if 'positive' in row else row[17:-4] for row in df['audio']]
    
    list_pos = [file for file in df['audio'] if 'positive' in file]
    list_neg = [file for file in df['audio'] if 'negative' in file]
    list_neu = [file for file in df['audio'] if 'neutral' in file]
    dict_ID = dict(zip(sorted(list(df.audio.unique())), list(range(0, len(list(df.audio.unique()))))))
    df.loc[:,variables[i_var]] = df.loc[:,variables[i_var]].apply(pd.to_numeric)
        
    audio_names = sorted(df['audio'].unique())
    df_data = df
    col_group = 'audio'
    label = variables[i_var]
    lst_groups = sorted(df['audio'].unique())
    lst_groups = [id_int[:-3] + ' ' + str(int(id_int[-2:])+1) for id_int in lst_groups]

          
    ax = fig.add_subplot(3,int(1), i_var + 1)
    df_plot = df_data.dropna(subset =[col_group,label], axis = 0).copy()
     
    lst_ala = []
    lst_col = []
    lst_boo = []
    lst_crit = []
    lst_color=['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet', 'darkturquoise']
     
    for idx, col in enumerate(sorted(df_plot[col_group].unique())):   
        if col in dict_critical[variables[i_var]]:
            lst_crit.append(idx)
        vals   = df_plot.loc[df_plot[col_group]==col,label].values
        
        lst_ala.append(vals)
        lst_col.append(col)
        if fwr_correction:   
            alpha = 1 - (0.05 / len(sorted(df_plot[col_group].unique())))
        else: 
            alpha = 1 - (0.05)
        if boot == 'mean':
            dict_b = pb.bootstrapping(vals,
                     numb_iterations = boot_size,
                     alpha = alpha,
                     as_dict = True,
                     func = 'mean')
            lst_boo.append(dict_b)
            
      
    df_boo = pd.DataFrame(lst_boo)
    
    eplot = ax.errorbar(x = np.array(df_boo.index),
                y = df_boo['mean'].values,
                yerr = abs((df_boo.loc[:,['lower','upper']].T-df_boo['mean'].values).values) ,
                marker = 'o',
                ls = '',
                capsize = 5,
                c = 'black') 
    
    eplot = ax.errorbar(x = np.array(df_boo.index[lst_crit]),
                y = df_boo['mean'][lst_crit].values,
                yerr = abs((df_boo.loc[lst_crit,['lower','upper']].T-df_boo['mean'][lst_crit].values).values) ,
                marker = 'o',
                ls = '',
                capsize = 5,
                #TODO!! add color if you want to highlight critical ones
                c = 'black')     
    
    ax.plot()
    if i_var + 1 == len(variables):
        ax.set_xticks(np.arange(0, len(lst_groups), 1))
        ax.set_xticklabels(lst_groups,fontsize=20, fontweight='bold', rotation=45, ha='right', c = 'black',fontname="Times New Roman" )
    else: 
        ax.set_xticks(np.arange(0, len(lst_groups), 1))
        ax.set_xticklabels([''] * len(lst_groups),fontsize=20, fontweight='bold', rotation=45, ha='right', c = 'black' )
    ax.set_title(titles[i_var], fontsize = 22, fontweight='bold',c = 'black', fontname="Times New Roman")
    ax.set_ylim([0, 100])
    ax.set_yticklabels(np.linspace(0,100,6), fontsize = 20, fontweight='bold',c = 'black',fontname="Times New Roman" )
    ax.set_ylabel('Scale (1 - 100)', fontsize = 20, fontweight='bold',c = 'black')
    df_boo.index = sorted(df_plot[col_group].unique())
    df_boo.index.rename(label, inplace = True)

    ax.axhline(np.mean(df_boo.loc[df_boo.index.isin(list_neg), 'mean']), c = dict_color['negative'],ls='--')
    ax.axhline(np.mean(df_boo.loc[df_boo.index.isin(list_neu), 'mean']), c=dict_color['neutral'], ls='--')
    ax.axhline(np.mean(df_boo.loc[df_boo.index.isin(list_pos), 'mean']), c = dict_color['positive'],ls='--')

    mpl.rc('font',family='Times New Roman', weight = 'bold')
    fig.legend([Line2D([0], [0], color=dict_color['negative'], lw=3, ls='--'),
              Line2D([0], [0], color=dict_color['neutral'], lw=3, ls='--'),
              Line2D([0], [0], color=dict_color['positive'], lw=3, ls='--')],
              ['NEGATIVE', 'NEUTRAL', 'POSITIVE'] ,fontsize='x-large')

    df_boo.to_csv(os.path.join(save_path, 'Bootstrapped_comparison_'+ titles[i_var] +'.csv'), index = True, header = (not os.path.exists(os.path.join(save_path, 'Bootstrapped_comparison_'+ label + '_'+ titles[i_var] +'.csv'))), mode = 'a', decimal=',', sep=';' )

fig.tight_layout(rect=[0,0,1,0.99])
fig.savefig(os.path.join(save_path, 'Bootstrapped.svg'))
plt.close()
        
    
for i_var, setting in enumerate(['valence_based', 'arousal_based', 'dominance_based']):
    save = os.path.join(save_path, setting)
    if not os.path.exists(save):
        os.makedirs(save)    
    # =============================================================================
    # Monte Carlo Simulation - Distribution 
    # =============================================================================
    #Selection of Critical Files
    # negative - neutral
    df = pd.read_csv(os.path.join(data, 'df_features.csv'), decimal=',', sep=';')
    condition_list = df.loc[:, ['condition']]
    cols = ['condition','audio','subj'] + [variables[i_var]]
    df = df.loc[:, cols]
    df['audio'] = ['positive_' + row[-6:-4] if 'positive' in row else row[17:-4] for row in df['audio']]
    df['audio'] = [row[:3] + ' ' + str(int(row[-2:])+1) for row in df['audio']]
    list_pos = [file for file in df['audio'] if 'pos' in file]
    list_neg = [file for file in df['audio'] if 'neg' in file]
    list_neu = [file for file in df['audio'] if 'neu' in file]
    dict_ID = dict(zip(sorted(list(df.audio.unique())), list(range(0, len(list(df.audio.unique()))))))
    df.loc[:,variables[i_var]] = df.loc[:,variables[i_var]].apply(pd.to_numeric)

    
    negative_files = df.loc[np.where(df.audio.isin(list_neg))[0], [variables[i_var], 'audio']]
    neutral_files = df.loc[np.where(df.audio.isin(list_neu))[0], [variables[i_var], 'audio']]
    positive_files = df.loc[np.where(df.audio.isin(list_pos))[0], [variables[i_var], 'audio']]
    
    if False:
        count = 0
        df_msc = pd.DataFrame(columns = ['Audio', 'Reference','Audio-Reference', 'Probability_errorous_labeling'])
        x_label = False
        for idx, cond in enumerate([negative_files, positive_files]):
            audio_fs = sorted(cond.audio.unique())
            for seed, audio_f in enumerate(audio_fs):
                if True: #audio_f not in dict_critical[variables[i_var]]
                    np.random.seed(seed)
                    mcs_f = np.random.choice(cond.loc[cond.audio == audio_f, variables[i_var]].values, n_simulation)
                    for ref_f in sorted(neutral_files.audio.unique()):
                        if True: #ref_f not in dict_critical[variables[i_var]]
                            count = count + 1
                            mcs_r = np.random.choice(neutral_files.loc[neutral_files.audio == ref_f, variables[i_var]].values, n_simulation)
                            df_msc.loc[count, 'Audio'] = audio_f
                            df_msc.loc[count, 'Reference'] = ref_f
                            df_msc.loc[count, 'Audio-Reference'] = audio_f + ' - ' + ref_f

                            if ('pos' in audio_f) and ('valence' in setting):
                                df_msc.loc[count, 'Probability_errorous_labeling'] = np.where((mcs_f - mcs_r) < 0)[0].shape[0] / n_simulation
                            elif ('neg' in audio_f) and ('valence' in setting):
                                df_msc.loc[count, 'Probability_errorous_labeling'] = np.where((mcs_r - mcs_f) < 0)[0].shape[0] / n_simulation
                            elif  ('arousal' in setting):
                                df_msc.loc[count, 'Probability_errorous_labeling'] = np.where((mcs_f - mcs_r) < 0)[0].shape[0] / n_simulation
                            elif ('dominance' in setting):
                                df_msc.loc[count, 'Probability_errorous_labeling'] = np.where((mcs_f - mcs_r) < 0)[0].shape[0] / n_simulation


        #df_critical = df_msc.loc[df_msc['Probability_errorous_labeling'] > 0.3]
        #df_critical.groupby(by = ['Audio'])['Audio'].count().sort_values(ascending=False)
        #df_critical.groupby(by = ['Reference'])['Reference'].count().sort_values(ascending=False)
        df_critical = df_msc.copy()
        df_critical = df_critical.sort_values(by = 'Probability_errorous_labeling', ascending=False)
        df_msc.to_csv(os.path.join(save, 'Probability_errorous_labeling_'+ setting +'.csv'), header = True, decimal=',', sep=';' )

        # =============================================================================
        # Full Figure
        # =============================================================================

        #df_critical = df_msc.copy()
        fig = plt.figure(figsize=(15, 12), facecolor='white')
        fig.suptitle('Audiofiles sorted in Probability of Erroneous '+ titles[i_var] +' Evaluation', fontsize = 30, fontweight='bold',fontname="Times New Roman")
        ax = fig.add_subplot(1,int(1), 1)

        my_cmap = plt.cm.get_cmap('YlOrRd')
        data_normalized = [x / max(df_critical['Probability_errorous_labeling'].values) for x in df_critical['Probability_errorous_labeling'].values]
        colors = my_cmap(data_normalized)
        sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0,max(df_critical['Probability_errorous_labeling'].values)))
        ax.bar(df_critical['Audio-Reference'].values, df_critical['Probability_errorous_labeling'].values, color = colors)
        #cbar = plt.colorbar(sm)
        ax.set_xticks(np.arange(0, len(df_critical['Probability_errorous_labeling'].values), 1))
        if x_label:
            ax.set_xticklabels(df_critical['Audio-Reference'].values,fontsize=20, fontweight='bold', rotation=45, ha='right', c = 'black',fontname="Times New Roman")
        if not x_label:
            ax.set_xticklabels([''] * len(df_critical['Audio-Reference'].values),fontsize=20, fontweight='bold', rotation=45, ha='right', c = 'black' )#

        ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 24, fontweight='bold',c = 'black',fontname="Times New Roman")
        ax.axhline(0.3, linestyle = 'dashed',color='black')
        proportion = np.round(len(df_critical.loc[df_critical['Probability_errorous_labeling'] > 0.3]) / count * 100, 2)
        ax.text(20,0.305, str(proportion) + ' % above threshold',fontsize=24, fontweight = 'bold',fontname="Times New Roman")

        #ax.set_xlabel('Compared Audiosegments', fontsize=14, fontweight = 'bold')
        ax.set_ylabel('Probability of Erroneous '+ titles[i_var] +' Rating in %', fontsize=24, fontweight = 'bold',fontname="Times New Roman")

        fig.subplots_adjust(bottom=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(save, 'Probability_Erroneous_Rating_Full.svg'))
        plt.close()

        # =============================================================================
        # Selected Plot
        # =============================================================================

        df_critical = df_msc.loc[df_msc['Probability_errorous_labeling'] > 0.2]
        df_critical = df_critical.sort_values(by = 'Probability_errorous_labeling', ascending=False)
        fig = plt.figure(figsize=(30, 12), facecolor='white')
        fig.suptitle('Critical Audiofiles with Probability above a 20 % Threshold', fontsize = 30, fontweight='bold',fontname="Times New Roman")  #' - '+ titles[i_var]
        ax = fig.add_subplot(1,int(1), 1)

        my_cmap = plt.cm.get_cmap('YlOrRd')
        data_normalized = [x / max(df_critical['Probability_errorous_labeling'].values) for x in df_critical['Probability_errorous_labeling'].values]
        colors = my_cmap(data_normalized)
        sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0,max(df_critical['Probability_errorous_labeling'].values)))
        ax.bar(df_critical['Audio-Reference'].values, df_critical['Probability_errorous_labeling'].values, color = colors, align='edge')
        #cbar = plt.colorbar(sm)
        ax.set_xticks(np.arange(0, len(df_critical['Probability_errorous_labeling'].values), 1))
        ax.set_xticklabels(df_critical['Audio-Reference'].values,fontsize=24, fontweight='bold', rotation=45, ha='right', c = 'black',fontname="Times New Roman" )#

        ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 24, fontweight='bold',c = 'black',fontname="Times New Roman")
        ax.axhline(0.2, linestyle = 'dashed',color='black')
        proportion = np.round(len(df_critical.loc[df_critical['Probability_errorous_labeling'] > 0.2]) / count * 100, 2)
        if setting == 'valence_based':
            ax.text(35,0.305, str(proportion) + ' % of pairs above 20 % - threshold',fontsize=30, fontweight = 'bold',fontname="Times New Roman")
        if setting == 'arousal_based':
            ax.text(26,0.305, str(proportion) + ' % of pairs above 20 % - threshold',fontsize=30, fontweight = 'bold',fontname="Times New Roman")
        if setting == 'dominance_based':
            ax.text(25,0.305, str(proportion) + ' % of pairs above 20 % - threshold',fontsize=30, fontweight = 'bold',fontname="Times New Roman")

        ax.set_ylabel('Probability of Erroneous '+ titles[i_var] +' Rating in %', fontsize=24, fontweight = 'bold',fontname="Times New Roman")

        fig.subplots_adjust(bottom=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(save, 'Probability_Critical_Audio.svg'))
        plt.close()

    
    # =============================================================================
    # Cohens d
    # =============================================================================
    
    dict_cohens = {}
    var = variables[i_var]
    
    df = pd.read_csv(os.path.join(data, 'df_features.csv'), decimal=',', sep=';')
    condition_list = df.loc[:, ['condition']]
    cols = ['condition','audio','subj'] + [variables[i_var]]
    df = df.loc[:, cols]
    df['audio'] = ['positive_' + row[-6:-4] if 'positive' in row else row[17:-4] for row in df['audio']]
    list_pos = [file for file in df['audio'] if 'positive' in file]
    list_neg = [file for file in df['audio'] if 'negative' in file]
    list_neu = [file for file in df['audio'] if 'neutral' in file]
    df.loc[:,variables[i_var]] = df.loc[:,variables[i_var]].apply(pd.to_numeric)
    
    df_group_mean = df.groupby(['audio', 'condition']).mean().reset_index()
    df_group_std = df.groupby(['audio', 'condition']).std().reset_index()
    for file in df_group_mean['audio'].unique():
        if 'negative' in file:
            con = 'negative'
        
        elif 'positive' in file:
            con = 'positive'
            
        elif 'neutral' in file:
            con = 'neutral'
            
        comp_con = [cc for cc in [i[:-3] for i in list(df_group_mean.audio.unique())] if cc != con]        
        print(file, comp_con)
    
        for i_comp, cc in enumerate(comp_con):
            list_comp = []
            for comp_file in df_group_mean[df_group_mean['condition'] == comp_con[i_comp]]['audio']:
                print(comp_file, cc)
                cohens_d = (df_group_mean.loc[df_group_mean['audio'] == file,var].values - df_group_mean.loc[df_group_mean['audio'] == comp_file,var].values) / (sqrt(((df_group_std.loc[df_group_std['audio'] == file,var].values ** 2) + (df_group_std.loc[df_group_std['audio'] == comp_file,var].values ** 2)) / 2))
                list_comp.append(cohens_d[0])
                print(cohens_d)
            if file not in dict_cohens.keys():
                dict_cohens[file] = {cc: np.mean(abs(np.array(list_comp)))}
            else: 
                dict_cohens[file][cc] = np.mean(abs(np.array(list_comp)))
    
    df_new = pd.DataFrame(columns = ['audio', 'negative', 'positive', 'neutral'])    
    for i_key, key in enumerate(dict_cohens.keys()):
        df_new.loc[i_key, 'audio'] = key
        df_new.loc[i_key, list(dict_cohens[key].keys())[0]] = dict_cohens[key][list(dict_cohens[key].keys())[0]]
        df_new.loc[i_key, list(dict_cohens[key].keys())[1]] = dict_cohens[key][list(dict_cohens[key].keys())[1]]
    
    fig, axes = plt.subplots(2,2, figsize=(18,12),sharex=False, sharey=True)
    axes = axes.flatten()    
    lst_color = [ '#E2001A', '#B1C800', '#1F82C0','#1F82C0']
    lst_alpha = [0.6,0.6,1,0.4]
    for i, (file, ax) in enumerate(zip([ ('negative', 'neutral'), ('positive',  'neutral'),
                         ('neutral', 'negative'), ('neutral', 'positive') ], axes)):
        ref = [n for n in df_new['audio'].unique() if file[0] in n]
        reference = df_new.loc[df_new['audio'].isin(ref)]  
        reference = reference.sort_values(by = file[1], ascending = False)
        reference['audio'] = [str(int(f[-2:]) + 1) for f in reference['audio']]
        ax.bar(reference['audio'], reference[file[1]], color = lst_color[i], alpha = lst_alpha[i])    
        ax.set_title(file[1] + ' audio segments as reference', fontsize=30, fontweight = 'bold',fontname="Times New Roman")  
        ax.set_xlabel(file[0] + ' audiosegments', fontsize=30, fontweight = 'bold',fontname="Times New Roman")
        if ax in [axes[0], axes[2]]:
            ax.set_ylabel('Average Cohens d', fontsize=26, fontweight = 'bold',fontname="Times New Roman")
        ax.tick_params(axis='x', labelsize=26)
        ax.tick_params(axis='y', labelsize=26)
    fig.suptitle('Discriminatory Power based on '+ titles[i_var] +' via the Cohens d', fontsize=34, fontweight = 'bold',fontname="Times New Roman") 
    # set the spacing between subplots
    fig.tight_layout(rect = [0, 0, 1, 0.98],h_pad=2.5)
    plt.show()
    # save the file 
    fig.savefig(os.path.join(save, 'Cohens_d_'+ var + '.svg'), bbox_inches='tight')
    plt.close(fig)
    num_cols = list(df_new.columns[1:])
    df_new[num_cols] = df_new[num_cols].apply(pd.to_numeric, errors='coerce')
    df_new.to_csv(os.path.join(save, 'Cohens_d_'+ var +'.csv'), index = False, header = True, decimal=',', sep=';' )
    
    # =============================================================================
    # Second Implementation Cohens d 
    # =============================================================================
    
    
    dict_cohens = {}
    var = variables[i_var]
    df_group_mean = df.groupby(['audio', 'condition']).mean().reset_index()
    
    for file in df_group_mean['audio'].unique():
        if 'negative' in file:
            con = 'negative'
        
        elif 'positive' in file:
            con = 'positive'
            
        elif 'neutral' in file:
            con = 'neutral'
            
        comp_con = [cc for cc in [i[:-3] for i in list(df_group_mean.audio.unique())] if cc != con]        
        print(file, comp_con)
    
        for i_comp, cc in enumerate(np.unique(comp_con)):
            print(cc)
            reference_data = df_group_mean[df_group_mean['condition'] == cc][var].values
            cohens_d = pg.compute_effsize(np.array(list(df_group_mean.loc[df_group_mean['audio'] == file][var].values) * len(reference_data)), df_group_mean[df_group_mean['condition'] == comp_con[i_comp]][var].values, paired=True, eftype='cohen')
            if file not in dict_cohens.keys():
                dict_cohens[file] = {cc: abs(cohens_d)}
            else: 
                dict_cohens[file][cc] = abs(cohens_d)
    
    df_new = pd.DataFrame(columns = ['audio', 'negative', 'positive', 'neutral'])    
    for i_key, key in enumerate(dict_cohens.keys()):
        df_new.loc[i_key, 'audio'] = key
        df_new.loc[i_key, list(dict_cohens[key].keys())[0]] = dict_cohens[key][list(dict_cohens[key].keys())[0]]
        df_new.loc[i_key, list(dict_cohens[key].keys())[1]] = dict_cohens[key][list(dict_cohens[key].keys())[1]]
    
    fig, axes = plt.subplots(2,2, figsize=(20,15),sharex=False, sharey=True)
    axes = axes.flatten()    
    lst_color = [ '#E2001A', '#B1C800', '#1F82C0','#1F82C0']
    lst_alpha = [0.6,0.6,1,0.6]
    for i, (file, ax) in enumerate(zip([ ('negative', 'neutral'), ('positive',  'neutral'),
                         ('neutral', 'negative'), ('neutral', 'positive') ], axes)):
        ref = [n for n in df_new['audio'].unique() if file[0] in n]
        reference = df_new.loc[df_new['audio'].isin(ref)]  
        reference = reference.sort_values(by = file[1], ascending = False)
        reference['audio'] = [f[-2:] for f in reference['audio']]
        ax.bar(reference['audio'], reference[file[1]], color = lst_color[i], alpha = lst_alpha[i])    
        ax.set_title(file[1] + ' Audiosegments as Reference', fontsize=18, fontweight = 'bold')  
        ax.set_xlabel(file[0] + ' Audiosegments', fontsize=14, fontweight = 'bold')
        ax.set_ylabel('Average Cohens d', fontsize=14, fontweight = 'bold')
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
    fig.suptitle('Discriminatory Power based on '+ titles[i_var] +' via the Cohens d', fontsize=24, fontweight = 'bold') 
    # set the spacing between subplots
    plt.tight_layout()
                
    # save the file 
    fig.savefig(os.path.join(save, 'Cohens_d_ping_'+ var + '.svg'))
    plt.close(fig)
    num_cols = list(df_new.columns[1:])
    df_new[num_cols] = df_new[num_cols].apply(pd.to_numeric, errors='coerce')
    df_new.to_csv(os.path.join(save, 'Cohens_d_ping_'+ var +'.csv'), index = False, header = True, decimal=',', sep=';' )