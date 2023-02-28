# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:16:36 2022

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
from scipy import stats
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
from statsmodels.stats.multitest import multipletests
import config_analysis

mpl.rc('font',family='Times New Roman', weight = 'bold')
lst_color = ['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet']
dict_color = {'positive': lst_color[2],
              'neutral' : lst_color[0],
              'negative' : lst_color[1]}

bids_root = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
data = os.path.join(bids_root, 'derivatives', 'ratings')
save = os.path.join(bids_root, 'derivatives', 'ratings')
if not os.path.exists(save):
    os.makedirs(save)

df = pd.read_csv(os.path.join(data, 'df_features.csv'), decimal=',', sep=';')
df['emotion'] = ['positive' if emo in ['Freude', '�berraschung'] else 'negative' if emo in ['Trauer','Angst', 'Ekel', 'Wut'] else emo for emo in list(df['emotion'])]
gew_neg = ['Angst', 'Ekel', 'Hass', 'Wut', 'Ent�uschung', 'Trauer','Verachtung','Scham', 'Schuld', 'Bereuen']
gew_pos = ['Interesse', 'Freude', 'Vergn�gen', 'Zufriedenheit','Belustigung', 'Mitgef�hl', 'Stolz',  'Erleichterung']
df['geneva_wheel'] = ['positive' if emo in gew_pos else 'negative' if emo in gew_neg else emo for emo in list(df['emotion'])]
list_geneva_wheel = df['geneva_wheel']
df = pd.get_dummies(df, columns = ['emotion', 'geneva_wheel'])
for i, strength_response in enumerate(list_geneva_wheel):
    df.loc[i, 'geneva_wheel_' + strength_response] = df.loc[i, 'geneva_wheel_' + strength_response] *  df.loc[i, 'strength']
    
cols = [col for col in df.columns if col not in ['Unnamed: 0']]
df = df.loc[:, cols].reset_index(drop = True)
df['audio'] = ['positive_' + row[-6:-4] if 'positive' in row else row[17:-4] for row in df['audio']]

if '_excluded' in save:
    df = df.loc[~df['audio'].isin(['neutral_02', 'neutral_04', 'neutral_07', 'neutral_09', 'positive_06', 'positive_07']),:] 

num_cols = [col for col in df.columns if col not in ['condition', 'audio', 'subj']]
df.loc[:,num_cols] = df.loc[:,num_cols].apply(pd.to_numeric)

    
variables = ['emotion_negative','emotion_positive','geneva_wheel_negative','geneva_wheel_positive', 'strength']
variables_2 = ['familiarity']
variables_3 = ['valence_std','arousal_std']

# =============================================================================
# Descriptives per Audiofile
# =============================================================================
titles = ['Negative Basic Emotion','Positive Basic Emotion', 'Negative GEW Emotion',  'Positive GEW Emotion', 'GEW Strength']
titles_2 = ['Familiarity']
titles_3 = ['Standard Deviation of the Valence Ratings Across Participants', 'Standard Deviation of the Arousal Ratings Across Participants']


if True:
    fig = plt.figure(figsize=(20, 10), facecolor='white')   
    fig.suptitle('Basic and GEW Emotion Indices', fontsize = 28, fontweight='bold')      
    
    for i_var, var in enumerate(variables):
        df_plot = df.copy()
        condition_list = df_plot.loc[:, ['condition']]
        cols = ['condition','audio','subj'] + [var]
        df_plot = df_plot.loc[:, cols]
        list_pos = [file for file in df_plot['audio'] if 'positive' in file]
        list_neg = [file for file in df_plot['audio'] if 'negative' in file]
        list_neu = [file for file in df_plot['audio'] if 'neutral' in file]
        dict_ID = dict(zip(sorted(list(df_plot.audio.unique())), list(range(0, len(list(df_plot.audio.unique()))))))
        df_plot.loc[:,variables[i_var]] = df_plot.loc[:,variables[i_var]].apply(pd.to_numeric)
            
        audio_names = sorted(df_plot['audio'].unique())
        col_group = 'audio'
        lst_groups = sorted(df_plot['audio'].unique())
        lst_groups = [id_int[:-3] + ' ' + str(int(id_int[-2:])+1) for id_int in lst_groups]
        fwr_correction = False
        boot_size = 5000
        boot = 'mean'
              
        ax = fig.add_subplot(len(variables),int(1), i_var + 1)
        df_plot = df_plot.dropna(subset =[col_group,var], axis = 0).copy()
         
        lst_ala = []
        lst_col = []
        lst_boo = []
        lst_crit = []
        lst_color=['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet', 'darkturquoise']
         
        for idx, col in enumerate(sorted(df_plot[col_group].unique())):   
            #if col in dict_critical[variables[i_var]]:
            #    lst_crit.append(idx)
            vals   = df_plot.loc[df_plot[col_group]==col,var].values
            
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
        
        #eplot = ax.errorbar(x = np.array(df_boo.index[lst_crit]),
        #            y = df_boo['mean'][lst_crit].values,
        #            yerr = abs((df_boo.loc[lst_crit,['lower','upper']].T-df_boo['mean'][lst_crit].values).values) ,
        #            marker = 'o',
        #            ls = '',
        #            capsize = 5,
        #            c = 'red')     
        ax.plot()
        if i_var + 1 == len(variables):
            ax.set_xticks(np.arange(0, len(lst_groups), 1))
            ax.set_xticklabels(lst_groups,fontsize=20, fontweight='bold', rotation=45, ha='right', c = 'black' )
        else: 
            ax.set_xticks(np.arange(0, len(lst_groups), 1))
            ax.set_xticklabels([''] * len(lst_groups),fontsize=20, fontweight='bold', rotation=45, ha='right', c = 'black' )
        ax.set_title(titles[i_var], fontsize = 24, fontweight='bold',c = 'black')
        
        if var not in variables[-3:]:
            ax.set_ylim([-0.2, 1.2])
            ax.set_yticks(np.linspace(-0.2,1.2,8))
            ax.set_yticklabels(['','',0.2,'',0.6,'',1.0,''], fontsize = 20, fontweight='bold',c = 'black')
                 
        elif var in variables[-3:]:
            ax.set_ylim([-1, 7])
            ax.set_yticks(np.linspace(-1,7,9))
            ax.set_yticklabels(['',0,'',2,'',4,'',6,''], fontsize = 20, fontweight='bold',c = 'black')
        df_boo.index = sorted(df_plot[col_group].unique())
        df_boo.index.rename(var, inplace = True)
          
        ax.axhline(np.mean(df_boo.loc[df_boo.index.isin(list_pos), 'mean']), c = dict_color['positive'],ls='--')
        ax.axhline(np.mean(df_boo.loc[df_boo.index.isin(list_neg), 'mean']), c = dict_color['negative'],ls='--')
        ax.axhline(np.mean(df_boo.loc[df_boo.index.isin(list_neu), 'mean']), c = dict_color['neutral'],ls='--')

        fig.legend([Line2D([0], [0], color=dict_color['negative'], lw=3, ls = 'dashed'),
                  Line2D([0], [0], color=dict_color['neutral'], lw=3, ls = 'dashed'),
                  Line2D([0], [0], color=dict_color['positive'], lw=3, ls = 'dashed')],
                  ['NEGATIVE', 'NEUTRAL', 'POSITIVE'] ,fontsize='xx-large')
        df_boo.to_csv(os.path.join(save, 'Bootstrapped_GEW_BASIC_comparison_'+ var +'.csv'), index = True, header = (not os.path.exists(os.path.join(save, 'Bootstrapped_GEW_BASIC_comparison_'+ var +'.csv'))), mode = 'a', decimal=',', sep=';' )
    fig.text(0.01, 0.5, 'Scales', ha='center', va='center', rotation='vertical', size=22)
    fig.tight_layout(rect=[0.01,0,1,0.99])
    plt.show()
    fig.savefig(os.path.join(save, 'Bootstrapped_GEW_BASIC.svg'))
    plt.close()
    
      
    # =============================================================================
    # Descriptives per Audiofile
    # =============================================================================
    
    fig = plt.figure(figsize=(20, 5), facecolor='white')   
    fig.suptitle('Familiarity of the Audio Sequences', fontsize = 24, fontweight='bold')      
    
    for i_var, var in enumerate(variables_2):
        df_plot = df.copy()
        condition_list = df_plot.loc[:, ['condition']]
        cols = ['condition','audio','subj'] + [var]
        df_plot = df_plot.loc[:, cols]
        list_pos = [file for file in df_plot['audio'] if 'positive' in file]
        list_neg = [file for file in df_plot['audio'] if 'negative' in file]
        list_neu = [file for file in df_plot['audio'] if 'neutral' in file]
        dict_ID = dict(zip(sorted(list(df_plot.audio.unique())), list(range(0, len(list(df_plot.audio.unique()))))))
        df_plot.loc[:,variables_2[i_var]] = df_plot.loc[:,variables_2[i_var]].apply(pd.to_numeric)
            
        audio_names = sorted(df_plot['audio'].unique())
        col_group = 'audio'
        lst_groups = sorted(df_plot['audio'].unique())
        lst_groups = [id_int[:-3] + ' ' + str(int(id_int[-2:])+1) for id_int in lst_groups]
        fwr_correction = False
        boot_size = 5000
        boot = 'mean'
              
        ax = fig.add_subplot(len(variables_2),int(1), i_var + 1)
        df_plot = df_plot.dropna(subset =[col_group,var], axis = 0).copy()
         
        lst_ala = []
        lst_col = []
        lst_boo = []
        lst_crit = []
        lst_color=['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet', 'darkturquoise']
         
        for idx, col in enumerate(sorted(df_plot[col_group].unique())):   
            vals   = df_plot.loc[df_plot[col_group]==col,var].values
            
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
        
        #eplot = ax.errorbar(x = np.array(df_boo.index[lst_crit]),
        #            y = df_boo['mean'][lst_crit].values,
        #            yerr = abs((df_boo.loc[lst_crit,['lower','upper']].T-df_boo['mean'][lst_crit].values).values) ,
        #            marker = 'o',
        #            ls = '',
        #            capsize = 5,
        #            c = 'red')     
        ax.plot()
        if i_var + 1 == len(variables_2):
            ax.set_xticks(np.arange(0, len(lst_groups), 1))
            ax.set_xticklabels(lst_groups,fontsize=20, fontweight='bold', rotation=45, ha='right', c = 'black' )
        else: 
            ax.set_xticks(np.arange(0, len(lst_groups), 1))
            ax.set_xticklabels([''] * len(lst_groups),fontsize=20, fontweight='bold', rotation=45, ha='right', c = 'black' )
        #ax.set_title(titles_2[i_var], fontsize = 14, fontweight='bold',c = 'black')
        
        
        ax.set_ylim([0, 100])
        ax.set_yticks(np.linspace(0, 100,11))
        ax.set_yticklabels([0,'',20,'',40,'',60,'',80, '', 100], fontsize = 20, fontweight='bold',c = 'black')
        ax.set_ylabel('Scale (1 - 100)', fontsize = 20, fontweight='bold',c = 'black')
        df_boo.index = sorted(df_plot[col_group].unique())
        df_boo.index.rename(var, inplace = True)
          
        ax.axhline(np.mean(df_boo.loc[df_boo.index.isin(list_pos), 'mean']), c = dict_color['positive'],ls='--')
        ax.axhline(np.mean(df_boo.loc[df_boo.index.isin(list_neg), 'mean']), c = dict_color['negative'],ls='--')
        ax.axhline(np.mean(df_boo.loc[df_boo.index.isin(list_neu), 'mean']), c = dict_color['neutral'],ls='--')
        fig.legend([Line2D([0], [0], color=dict_color['negative'], lw=3, ls = 'dashed'),
                  Line2D([0], [0], color=dict_color['neutral'], lw=3, ls = 'dashed'),
                  Line2D([0], [0], color=dict_color['positive'], lw=3, ls = 'dashed')],
                  ['NEGATIVE', 'NEUTRAL', 'POSITIVE'] ,fontsize='xx-large')
        df_boo.to_csv(os.path.join(save, 'Bootstrapped_fam_comparison_'+ var +'.csv'), index = True, header = (not os.path.exists(os.path.join(save, 'Bootstrapped_fam_comparison_'+ var +'.csv'))), mode = 'a', decimal=',', sep=';' )
    
    fig.tight_layout(rect=[0,0,1,0.99])
    fig.savefig(os.path.join(save, 'Bootstrapped_fam.svg'))
    plt.close()
    
# =============================================================================
# Variability
# =============================================================================
    fig = plt.figure(figsize=(20, 10), facecolor='white')   
    #fig.suptitle('Variability of the Continuous Ratings per Audio Sequence', fontsize = 28, fontweight='bold')      
    
    for i_var, var in enumerate(variables_3):
        df_plot = df.copy()
        condition_list = df_plot.loc[:, ['condition']]
        cols = ['condition','audio','subj'] + [var]
        df_plot = df_plot.loc[:, cols]
        list_pos = [file for file in df_plot['audio'] if 'positive' in file]
        list_neg = [file for file in df_plot['audio'] if 'negative' in file]
        list_neu = [file for file in df_plot['audio'] if 'neutral' in file]
        dict_ID = dict(zip(sorted(list(df_plot.audio.unique())), list(range(0, len(list(df_plot.audio.unique()))))))
        df_plot.loc[:,variables_3[i_var]] = df_plot.loc[:,variables_3[i_var]].apply(pd.to_numeric)
            
        audio_names = sorted(df_plot['audio'].unique())
        col_group = 'audio'
        lst_groups = sorted(df_plot['audio'].unique())
        lst_groups = [id_int[:-3] + ' ' + str(int(id_int[-2:])+1) for id_int in lst_groups]
        fwr_correction = False
        boot_size = 5000
        boot = 'mean'
              
        ax = fig.add_subplot(len(variables_3),int(1), i_var + 1)
        df_plot = df_plot.dropna(subset =[col_group,var], axis = 0).copy()
         
        lst_ala = []
        lst_col = []
        lst_boo = []
        lst_crit = []
        lst_color=['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet', 'darkturquoise']
         
        for idx, col in enumerate(sorted(df_plot[col_group].unique())):   
            vals   = df_plot.loc[df_plot[col_group]==col,var].values
            
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
        
        #eplot = ax.errorbar(x = np.array(df_boo.index[lst_crit]),
        #            y = df_boo['mean'][lst_crit].values,
        #            yerr = abs((df_boo.loc[lst_crit,['lower','upper']].T-df_boo['mean'][lst_crit].values).values) ,
        #            marker = 'o',
        #            ls = '',
        #            capsize = 5,
        #            c = 'red')     
        ax.plot()
        if i_var + 1 == len(variables_3):
            ax.set_xticks(np.arange(0, len(lst_groups), 1))
            ax.set_xticklabels(lst_groups,fontsize=20, fontweight='bold', rotation=45, ha='right', c = 'black' )
        else: 
            ax.set_xticks(np.arange(0, len(lst_groups), 1))
            ax.set_xticklabels([''] * len(lst_groups),fontsize=20, fontweight='bold', rotation=45, ha='right', c = 'black' )
        ax.set_title(titles_3[i_var], fontsize = 24, fontweight='bold',c = 'black')
        
        
        ax.set_ylim([0, 25])
        ax.set_yticks(np.linspace(0, 25,6))
        ax.set_yticklabels([0,5,10,15,20,25], fontsize = 20, fontweight='bold',c = 'black')
        ax.set_ylabel('SD', fontsize = 24, fontweight='bold',c = 'black')
        df_boo.index = sorted(df_plot[col_group].unique())
        df_boo.index.rename(var, inplace = True)
          
        ax.axhline(np.mean(df_boo.loc[df_boo.index.isin(list_pos), 'mean']), c = dict_color['positive'],ls='--')
        ax.axhline(np.mean(df_boo.loc[df_boo.index.isin(list_neg), 'mean']), c = dict_color['negative'],ls='--')
        ax.axhline(np.mean(df_boo.loc[df_boo.index.isin(list_neu), 'mean']), c = dict_color['neutral'],ls='--')
        fig.legend([Line2D([0], [0], color=dict_color['negative'], lw=3, ls = 'dashed'),
                  Line2D([0], [0], color=dict_color['neutral'], lw=3, ls = 'dashed'),
                  Line2D([0], [0], color=dict_color['positive'], lw=3, ls = 'dashed')],
                  ['NEGATIVE', 'NEUTRAL', 'POSITIVE'] ,fontsize='xx-large')
        df_boo.to_csv(os.path.join(save, 'Bootstrapped_var_comparison_'+ var +'.csv'), index = True, header = (not os.path.exists(os.path.join(save, 'Bootstrapped_var_comparison_'+ var +'.csv'))), mode = 'a', decimal=',', sep=';' )
    fig.tight_layout(rect=[0,0,1,0.99])
    plt.show()
    fig.savefig(os.path.join(save, 'Bootstrapped_variability.svg'))
    plt.close()

# =============================================================================
# STATS 
# =============================================================================
variables =  ['valence_mean',  'arousal_mean', 'dominance'] + variables_3 + variables_2 +  variables
save_path = os.path.join(save, 'stats')
if not os.path.exists(save_path):
    os.makedirs(save_path)
# =============================================================================
# Over Conditions
# =============================================================================
if False:
    for var in variables:
        print(var)
        df_long = df.loc[:, ['subj','condition', var]]
        
    
        df_long.dropna(axis = 0, inplace =True)
        df_long[var] = df_long[var].astype(float)
        
        for i_cond, cond in enumerate(df_long['condition'].unique()):
            df_descriptive = df_long.loc[df_long['condition'] == cond, var].describe()
            df_descriptive.to_csv(os.path.join(save_path , 'Descriptives_' + cond + '_' + var + '.csv'), header = (not os.path.exists(os.path.join(save_path , 'Descriptives_' + cond + '_' + var + '.csv'))), index = True, sep =';', decimal=',', mode = 'a')
    
        # =============================================================================
        # Assumptions NV Shapiro Wilk and Spherizity Mauchly
        # =============================================================================
        list_violations = []
        sp_results = []
    
        sp = stats.shapiro(df_long[var])
        sp_results.append(sp)
        p = stats.shapiro(df_long[var])[1]
        if p < 0.05:
            print('ND violated ', var)
            list_violations.append(var)
        print(sp_results)
        
        for i_cond, cond in enumerate(df_long['condition'].unique()): 
            sp = stats.shapiro(df_long.loc[df_long['condition'] == cond, var])
            sp_results.append(sp)
            p = stats.shapiro(df_long[var])[1]
            if p < 0.05:
                print('ND violated ', var, cond)
                list_violations.append(var +'_' + str(cond))
        print(sp_results)
        df_groupby = df_long.groupby(['subj','condition']).mean().reset_index()
        df_wide = df_groupby.pivot(index='subj', columns='condition', values=var)
    
    
        # =============================================================================
        # Anova
        # =============================================================================
        aov = pg.rm_anova(dv=var, 
              within='condition',
              subject='subj', data=df_long, 
              correction = True,
              detailed=True)
    
    
        # =============================================================================
        # Non-parametric friedman chi
        # =============================================================================
        
        Var1 = df_wide.iloc[:,0].values
        Var2 = df_wide.iloc[:,1].values
        Var3 = df_wide.iloc[:,2].values
    
        stat, p = friedmanchisquare(Var1, Var2, Var3)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
    
        # =============================================================================
        # Assumptions Post-Hoc 
        # =============================================================================
        df_single_tests = pd.DataFrame()
           
        pairs_name = [['negative', 'neutral'], 
                     ['negative', 'positive'], 
                     ['positive', 'neutral']]
        names = list()  
        for name_pair in pairs_name:
            names.append(name_pair[0] + '_vs._' + name_pair[1])
                
        # =============================================================================
        # Post Hoc     
        # =============================================================================
        
        non_para_wilcoxon = []
        non_param_pvals = []
        
        t_tests = []
        pvals = []
        
        for pair in pairs_name: 
            try:
                stat_w, p_w = wilcoxon(df_wide[pair[0]] , df_wide[pair[1]])
                print('Statistics=%.3f, p=%.3f' % (stat_w, p_w))
                non_para_wilcoxon.append((pair,'Statistics=%.3f, p=%.3f' % (stat_w, p_w)))
                non_param_pvals.append(p_w)
            except: 
                print('Difference is zero', df_wide[pair[0]] - df_wide[pair[1]])
            stat_t, p_t = stats.ttest_rel(df_wide[pair[0]] , df_wide[pair[1]])
            print('Statistics=%.3f, p=%.3f' % (stat_t, p))
            t_tests.append((pair,'Statistics=%.3f, p=%.3f' % (stat_t, p_t)))
            pvals.append(p_t)
        
        # =============================================================================
        # Multiple testing
        # =============================================================================
        corrected_pvals = multipletests(non_param_pvals, alpha = 0.05, method = 'fdr_bh', is_sorted = False) 
        non_para_wilcoxon.append(corrected_pvals)
        
        corrected_pvals = multipletests(pvals, alpha = 0.05, method = 'fdr_bh', is_sorted = False) 
        t_tests.append(corrected_pvals)   
    
    
        aov.index.rename(var, inplace = True)
        aov.to_csv(os.path.join(save_path,  + 'ANOVA_results.csv'), index = True, header = True, mode = 'a', decimal=',', sep=';' )
         
        df_non_para = pd.DataFrame({'stats': [stat],
                        'pvalue' : [p]})
        df_non_para.index.rename(var, inplace = True)
        df_non_para.to_csv(save_path + '/Non_para_results.csv', sep=';', decimal=',', mode = 'a')
     
        df_post_hoc = pd.DataFrame({'non_para': [non_para_wilcoxon],
                        'ttest' : [t_tests]})
        df_post_hoc.index.rename(var, inplace = True)
        df_post_hoc.to_csv(os.path.join(save_path,  + 'Post-Hoc_results.csv'), sep=';', decimal=',', mode = 'a')
    
        
    
        df_count_sample = df_long.groupby(['subj', 'condition']).count()[var].reset_index()              
        df_count_sample.index.rename(var, inplace = True)
        df_count_sample.to_csv(os.path.join(save_path, 'Sample_count.csv'), sep=';', decimal=',', mode = 'a', header=(not os.path.exists(os.path.join(save_path, 'Sample_count.csv'))))
             
# =============================================================================
# Stack dataframe to short format necessary for rmANOVA and some visualizations
# =============================================================================
title = 'Comparison between Conditions'
subtitle = ['Valence',
             'Arousal',
             'Dominance',
             'Variability in Valence',
             'Variability in Arousal',
             'Familiarity',
             'Negative Basic Emotion',
             'Positive Basic Emotion',
             'Negative GEW Emotion',
             'Positive GEW Emotion',
             'GEW Intensity/Strength']



fig, df_CI = pb.plot_boxplots_stats_diff(df['condition'].unique(), 
       df,
       col_group="condition",
       labels= variables, 
       boot = 'mean',  
       boot_size = 5000,
       title=title, 
       subtitle = subtitle,
       lst_color=['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet', 'deeppink', 'cyan', 'green'],
       save_path = save_path, 
       fwr_correction = True)

for end in ['.svg', '.tif']:
    plt.savefig(os.path.join(save_path, 'Bootstrapped_results' + end))
plt.close()

df_CI.to_csv(os.path.join(save_path, 'Bootstrapped_results_stats.csv'), header = True, sep=';', decimal=',')
