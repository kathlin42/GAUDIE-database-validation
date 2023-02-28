# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 15:41:58 2021

@author: hirning
"""
# =============================================================================
# Import Packages
# =============================================================================
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random

# =============================================================================
# Helper Functions
# =============================================================================

def percentiles(lst_vals,alpha,func = 'mean'):
    
    lower = np.percentile(np.array(lst_vals), ((1.0-alpha)/2.0) * 100, axis = 0)
    upper = np.percentile(lst_vals, (alpha+((1.0-alpha)/2.0)) * 100, axis = 0)
    if func == 'mean':
        mean  = np.mean(lst_vals, axis = 0)
    elif func == 'median':
        mean  = np.median(lst_vals, axis = 0)
    return lower, mean, upper

def bootstrapping(input_sample,
                 sample_size = None,
                 numb_iterations = 4000,
                 alpha =0.95,
                 plot_hist =False,
                 as_dict = True,
                 func = 'mean'): #mean, median
    
    if sample_size==None:
        sample_size = len(input_sample)
        
    lst_means = []
    
    # ---------- Bootstrapping ------------------------------------------------
    
    print('\nBootstrapping with {} iterations and alpha: {}'.format(numb_iterations,alpha))
    for i in range(numb_iterations):
        try:
            random.seed(i)
            re_sampled = random.choices(input_sample.values, k=sample_size)
        except:
            random.seed(i)
            re_sampled = random.choices(input_sample, k=sample_size)
            
        if func == 'mean':
            lst_means.append(np.nanmean(np.array(re_sampled), axis=0))
        elif func == 'median': 
            lst_means.append(np.median(np.array(re_sampled), axis=0))
        #lst_means.append(np.median(np.array(re_sampled), axis=0))
        
    
    # ---------- Konfidenzintervall -------------------------------------------
    
    lower, mean, upper = percentiles(lst_means,alpha)
    
    dict_return = {'lower': lower,'mean':mean,'upper':upper}
    
    # ---------- Visulisierung ------------------------------------------------   
    
    if plot_hist:
        plt.hist(lst_means)
    
    # ---------- RETURN -------------------------------------------------------    
    
    if as_dict:
        return dict_return
    else:
        return mean, np.array([np.abs(lower-mean),(upper-mean)])



def plot_boxplots_plus_ref(lst_groups,df_data,col_group,labels, boot = 'median',  
               boot_size = 5000,title='', lst_color=None, save_path = None, refs = None,  fwr_correction = True):
    
       
    fig = plt.figure(figsize=(10, 8), facecolor='white')   
    fig.suptitle(title, fontsize = 20, fontweight='bold')
            
    for i, label in enumerate(labels):
        print(label)
        ax = fig.add_subplot(len(labels),1, i + 1)
        df_plot = df_data.dropna(subset =[label,col_group], axis = 0).copy()
 
        lst_ala = []
        lst_col = []
        lst_boo = []
           
        for col in lst_groups:    
            vals   = df_plot.loc[df_plot[col_group]==col,label].values
           
            lst_ala.append(vals)
            lst_col.append(col)
            if fwr_correction:   
                alpha = 1 - (0.05 / len(lst_groups))
            else: 
                alpha = 1 - (0.05)
            if boot == 'mean':
                dict_b = bootstrapping(vals,
                         numb_iterations = boot_size,
                         alpha = alpha,
                         as_dict = True,
                         func = 'mean')
                lst_boo.append(dict_b)
        if boot == 'mean':
            df_boo = pd.DataFrame(lst_boo)
            m      = df_data[label].mean()
        else:
            m      = df_data[label].median()

        df_boo = pd.DataFrame(lst_boo)
        
        bplot = ax.boxplot(lst_ala,
                              notch      = True,
                              usermedians= df_boo['mean'].values,
                              conf_intervals=df_boo.loc[:,['lower','upper']].values,
                              showfliers = False,
                              whis       = [5,95],
                              labels     = lst_col,
                              patch_artist=True)
        
        if lst_color is not None:
            for box, color in zip(bplot['boxes'], lst_color):
                box.set(color=color)
                box.set(facecolor=color, edgecolor='black') #, alpha=0.7
                
        ax.plot()
        ax.set_xticklabels(['']*len(lst_groups),fontsize=20, fontweight='bold', rotation=45, ha='right', c = 'black' )#
        ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 20, fontweight='bold')
        if 'BIG 5 - Short Version' in label:
            ax.set_ylabel('Scale (1 - 5)',fontsize=22, fontweight='bold', c = 'black' )
        if 'Well-Being Scales' in label :
            ax.set_ylabel('Normalized Scale (0 - 1)',fontsize=22, fontweight='bold', c = 'black' )
        #ax.set_xticklabels(lst_groups, rotation=45, ha='right')
        df_boo.index = lst_groups
        df_boo.index.rename(label, inplace = True)
        if 'Face' in title:
            save = os.path.join(save_path, 'Bootstrapped_comparison_Face.csv')
        elif 'NBack' in title:
            save = os.path.join(save_path, 'Bootstrapped_comparison_NBack.csv') 
        else:
            save = os.path.join(save_path, 'Bootstrapped_comparison_'+ label +'.csv')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #df_boo.to_csv(save, index = True, header = (not os.path.exists(save)), mode = 'a', decimal=',', sep=';' )
    
    ax.set_xticklabels(lst_groups,fontsize=20, fontweight='bold', rotation=45, ha='right', c = 'black' )#    
    if refs is not None: 
        for key, x_pos in zip(lst_groups, ax.get_xticks()-0.1):
            ax.text(x_pos, refs[key], '+', size = 30, color= 'blue')

    fig.tight_layout()# 
    return fig, df_boo
    
    
def plot_boxplots(lst_groups,df_data,col_group,labels, boot = 'median',  
               boot_size = 5000,title='', lst_color=None, save_path = None, fwr_correction = True):
    
       
    fig = plt.figure(figsize=(15, 10), facecolor='white')   
    fig.suptitle(title, fontsize = 20, fontweight='bold')
            
    for i, label in enumerate(labels):
        print(label)
        ax = fig.add_subplot(len(labels),1, i + 1)
        df_plot = df_data.dropna(subset =[label,col_group], axis = 0).copy()
 
        lst_ala = []
        lst_col = []
        lst_boo = []
           
        for col in lst_groups:    
            vals   = df_plot.loc[df_plot[col_group]==col,label].values
           
            lst_ala.append(vals)
            lst_col.append(col)
            if fwr_correction:   
                alpha = 1 - (0.05 / len(lst_groups))
            else: 
                alpha = 1 - (0.05)
            if boot == 'mean':
                dict_b = bootstrapping(vals,
                         numb_iterations = boot_size,
                         alpha = alpha,
                         as_dict = True,
                         func = 'mean')
                lst_boo.append(dict_b)
        if boot == 'mean':
            df_boo = pd.DataFrame(lst_boo)
            m      = df_data[label].mean()
        else:
            m      = df_data[label].median()

        df_boo = pd.DataFrame(lst_boo)
        
        bplot = ax.boxplot(lst_ala,
                              notch      = True,
                              usermedians= df_boo['mean'].values,
                              conf_intervals=df_boo.loc[:,['lower','upper']].values,
                              showfliers = False,
                              whis       = [5,95],
                              labels     = lst_col)
        
        if lst_color is not None:
            for box, color in zip(bplot['boxes'], lst_color):
                box.set(color=color)
        ax.plot(ax.axis()[0:2],
            [m]*2,
            c = 'blue',
            ls='--') 
           
        ax.plot()
        ax.set_xticklabels(['']*len(lst_groups),fontsize=14, fontweight='bold', rotation=45, ha='right', c = 'black' )#
        ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 11, fontweight='bold')
        ax.set_title(label, fontsize = 12, fontweight='bold')
        #ax.set_xticklabels(lst_groups, rotation=45, ha='right')
        df_boo.index = lst_groups
        df_boo.index.rename(label, inplace = True)
        if ('Audiofiles' in title) and (label in ['prob_negative', 'prob_neutral','prob_positive']): 
            ax.set_ylabel('FPC', fontsize = 11, fontweight='bold')
            if label == 'prob_negative':     
                ax.set_title('Probability to belong to the Negative Condition', fontsize = 12, fontweight='bold')
            if label == 'prob_positive':     
                ax.set_title('Probability to belong to the Positive Condition', fontsize = 12, fontweight='bold')
            if label == 'prob_neutral':     
                ax.set_title('Probability to belong to the Neutral Condition', fontsize = 12, fontweight='bold')

        if 'Face' in title:
            save = os.path.join(save_path, 'Bootstrapped_comparison_Face.csv')
        elif 'NBack' in title:
            save = os.path.join(save_path, 'Bootstrapped_comparison_NBack.csv') 
        else:
            save = os.path.join(save_path, 'Bootstrapped_comparison_'+ label +'.csv')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #df_boo.to_csv(save, index = True, header = (not os.path.exists(save)), mode = 'a', decimal=',', sep=';' )
    
    ax.set_xticklabels(lst_groups,fontsize=14, fontweight='bold', rotation=45, ha='right', c = 'black' )#    
    try:
        fig.legend([Line2D([0], [0], color=lst_color[0], lw=3), 
                    Line2D([0], [0], color=lst_color[1], lw=3)], 
                  [lst_groups[0], 
                   lst_groups[1]], 
                  loc ='upper right', 
                  facecolor='white',
                  fontsize='x-large',
                  ncol=2)
    except: 
        print('No colors defined')
    fig.tight_layout() 
    return  fig, df_boo


def plot_boxplots_stats(lst_groups,df_data,col_group,labels, boot = 'median',  
               boot_size = 5000,title='',subtitle =None, lst_color=None, save_path = None, fwr_correction = True):
       
    fig = plt.figure(figsize=(15, 10), facecolor='white')   
    fig.suptitle(title, fontsize = 20, fontweight='bold')
    df_full = pd.DataFrame()    
    for i, label in enumerate(labels):
        print(label)
        ax = fig.add_subplot(int(np.ceil(len(labels)/3)),3, i + 1)
        df_plot = df_data.dropna(subset =[label,col_group], axis = 0).copy()
 
        lst_ala = []
        lst_col = []
        lst_boo = []
           
        for col in lst_groups:    
            vals   = df_plot.loc[df_plot[col_group]==col,label].values
           
            lst_ala.append(vals)
            lst_col.append(col)
            if fwr_correction:   
                alpha = 1 - (0.05 / len(lst_groups))
            else: 
                alpha = 1 - (0.05)
            if boot == 'mean':
                dict_b = bootstrapping(vals,
                         numb_iterations = boot_size,
                         alpha = alpha,
                         as_dict = True,
                         func = 'mean')
                lst_boo.append(dict_b)
        if boot == 'mean':
            df_boo = pd.DataFrame(lst_boo)
            m      = df_data[label].mean()
        else:
            m      = df_data[label].median()

        df_boo = pd.DataFrame(lst_boo)
        
        eplot = ax.errorbar(x =0,
                            y = df_boo['mean'].values[0],
                            yerr = abs((df_boo.loc[0,['lower','upper']].T-df_boo['mean'][0]).values).reshape(2, 1),
                            marker = 'o',
                            ls = '',
                            capsize = 5,
                            c = lst_color[0])

        eplot = ax.errorbar(x = 1,
                            y = df_boo['mean'].values[1],
                            yerr = abs((df_boo.loc[1,['lower','upper']].T-df_boo['mean'][1]).values).reshape(2, 1) ,
                            marker = 'o',
                            ls = '',
                            capsize = 5,
                            c = lst_color[2])

        eplot = ax.errorbar(x = 2,
                            y = df_boo['mean'].values[2],
                            yerr = abs((df_boo.loc[2,['lower','upper']].T-df_boo['mean'][2]).values).reshape(2, 1),
                            marker = 'o',
                            ls = '',
                            capsize = 5,
                            c = lst_color[1])

        ax.plot()
        
        ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 11, fontweight='bold')
        if subtitle == None:
            ax.set_title(label, fontsize = 12, fontweight='bold')
        else:
             ax.set_title(subtitle[i], fontsize = 12, fontweight='bold')
        #ax.set_xticklabels(lst_groups, rotation=45, ha='right')
        df_boo.index = lst_groups
        #df_boo.index.rename(label, inplace = True)
        if ('Audiofiles' in title) and (label in ['prob_negative', 'prob_neutral','prob_positive']): 
            ax.set_ylabel('FPC', fontsize = 11, fontweight='bold')
            if label == 'prob_negative':     
                ax.set_title('Probability to belong to the Negative Condition', fontsize = 12, fontweight='bold')
            if label == 'prob_positive':     
                ax.set_title('Probability to belong to the Positive Condition', fontsize = 12, fontweight='bold')
            if label == 'prob_neutral':     
                ax.set_title('Probability to belong to the Neutral Condition', fontsize = 12, fontweight='bold')

        if 'Face' in title:
            save = os.path.join(save_path, 'Bootstrapped_comparison_Face.csv')
        elif 'NBack' in title:
            save = os.path.join(save_path, 'Bootstrapped_comparison_NBack.csv') 
        else:
            save = os.path.join(save_path, 'Bootstrapped_comparison_'+ label +'.csv')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df_boo['variable'] = label
        df_full = df_full.append(df_boo)
        
        if label in labels[-2:]:
            #ax.set_xlim([0, 2])
            ax.set_xticks(np.linspace(0,2,3))
            ax.set_xticklabels(lst_groups, fontsize = 14, rotation = 45, fontweight='bold',c = 'black')
        else: 
            ax.set_xticks(np.linspace(0,2,3))
            ax.set_xticklabels(['']* len(lst_groups), fontsize = 14, rotation = 45, fontweight='bold',c = 'black')

    ax = fig.add_subplot(int(np.ceil(len(labels)/3)),3, i + 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.axis('off')
    ax.legend([Line2D([0], [0], color=lst_color[0], lw=3), 
                Line2D([0], [0], color=lst_color[2], lw=3), 
                Line2D([0], [0], color=lst_color[1], lw=3)], 
              [lst_groups[0], 
               lst_groups[1],
               lst_groups[2]], 
              loc ='upper left', 
              facecolor='white',
              fontsize='x-large')
    fig.tight_layout() 
    return  fig, df_full


def plot_boxplots_stats_diff(lst_groups,df_data,col_group,labels, boot = 'median',  
               boot_size = 5000,title='',subtitle =None, lst_color=None, save_path = None, fwr_correction = True):
    
    #lst_groups = df['condition'].unique() 
    #df_data = df
    #col_group="condition"
    #labels= variables
    #boot = 'mean'
    #boot_size = 5000
    #title=title
    #subtitle = subtitle
    #lst_color=['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet', 'deeppink', 'cyan', 'green']
    #save_path = save_path 
    #fwr_correction = True
    
    
    fig = plt.figure(figsize=(15, 10), facecolor='white')   
    fig.suptitle(title, fontsize = 20, fontweight='bold')
    df_full = pd.DataFrame()    
    df_diff = pd.DataFrame(columns = ['subj', 'condition'] + labels)
    for subj in range(0,len(df_data['subj'].unique())):
        for cond_pair in [['negative', 'neutral'], ['positive', 'neutral'], ['negative', 'positive']]:
            df_temp = pd.DataFrame(columns = ['subj', 'condition'] + labels)
            df_subj = df_data.loc[(df_data['subj']==subj) & (df_data['condition'].isin(cond_pair))]
            print(df_subj)
            for variable in labels:
                c1 = df_subj.loc[df_subj['condition'] ==cond_pair[0], variable].values
                c1 = np.mean(c1[np.logical_not(np.isnan(c1))])
                c2 = df_subj.loc[df_subj['condition'] ==cond_pair[1], variable].values
                c2 = np.mean(c2[np.logical_not(np.isnan(c2))])
                df_temp.loc[0,'subj'] = subj
                df_temp.loc[0,'condition'] = str(cond_pair[0]) + ' - ' + str(cond_pair[1])
                df_temp.loc[df_temp['subj']==subj, variable] = c1 - c2
          
            df_diff = pd.concat([df_diff, df_temp], ignore_index=True)
            
        
    for i, label in enumerate(labels):
        print(label)
        ax = fig.add_subplot(int(np.ceil(len(labels)/3)),3, i + 1)
        ax.axhline(0, c = 'black',ls='--')
        df_plot = df_diff.dropna(subset =[label,col_group], axis = 0).copy()
 

        lst_ala = []
        lst_col = []
        lst_boo = []
           
        for col in df_plot['condition'].unique():    
            vals   = df_plot.loc[df_plot[col_group]==col,label].values
           
            lst_ala.append(vals)
            lst_col.append(col)
            if fwr_correction:   
                alpha = 1 - (0.05 / len(lst_groups))
            else: 
                alpha = 1 - (0.05)
            if boot == 'mean':
                dict_b = bootstrapping(vals,
                         numb_iterations = boot_size,
                         alpha = alpha,
                         as_dict = True,
                         func = 'mean')
                lst_boo.append(dict_b)
        if boot == 'mean':
            df_boo = pd.DataFrame(lst_boo)
            m      = df_data[label].mean()
        else:
            m      = df_data[label].median()

        df_boo = pd.DataFrame(lst_boo)
        
        eplot = ax.errorbar(x =0,
                            y = df_boo['mean'].values[0],
                            yerr = abs((df_boo.loc[0,['lower','upper']].T-df_boo['mean'][0]).values).reshape(2, 1),
                            marker = 'o',
                            ls = '',
                            capsize = 5,
                            c = 'grey')

        eplot = ax.errorbar(x = 1,
                            y = df_boo['mean'].values[1],
                            yerr = abs((df_boo.loc[1,['lower','upper']].T-df_boo['mean'][1]).values).reshape(2, 1) ,
                            marker = 'o',
                            ls = '',
                            capsize = 5,
                            c = 'grey')

        eplot = ax.errorbar(x = 2,
                            y = df_boo['mean'].values[2],
                            yerr = abs((df_boo.loc[2,['lower','upper']].T-df_boo['mean'][2]).values).reshape(2, 1),
                            marker = 'o',
                            ls = '',
                            capsize = 5,
                            c = 'grey')

      
        ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 11, fontweight='bold')
        if subtitle == None:
            ax.set_title(label, fontsize = 12, fontweight='bold')
        else:
            ax.set_title(subtitle[i], fontsize = 12, fontweight='bold')
        #ax.set_xticklabels(lst_groups, rotation=45, ha='right')
        df_boo.index = lst_groups
        #df_boo.index.rename(label, inplace = True)
        if ('Audiofiles' in title) and (label in ['prob_negative', 'prob_neutral','prob_positive']): 
            ax.set_ylabel('FPC', fontsize = 11, fontweight='bold')
            if label == 'prob_negative':     
                ax.set_title('Probability to belong to the Negative Condition', fontsize = 12, fontweight='bold')
            if label == 'prob_positive':     
                ax.set_title('Probability to belong to the Positive Condition', fontsize = 12, fontweight='bold')
            if label == 'prob_neutral':     
                ax.set_title('Probability to belong to the Neutral Condition', fontsize = 12, fontweight='bold')

        if 'Face' in title:
            save = os.path.join(save_path, 'Bootstrapped_comparison_Face.csv')
        elif 'NBack' in title:
            save = os.path.join(save_path, 'Bootstrapped_comparison_NBack.csv') 
        else:
            save = os.path.join(save_path, 'Bootstrapped_comparison_'+ label +'.csv')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df_boo['variable'] = label
        df_full = df_full.append(df_boo)
        
        if label in labels[-2:]:
            #ax.set_xlim([0, 2])
            ax.set_xticks(np.linspace(0,2,3))
            ax.set_xticklabels(df_plot['condition'].unique(), fontsize = 14, rotation = 45, fontweight='bold',c = 'black')
        else: 
            ax.set_xticks(np.linspace(0,2,3))
            ax.set_xticklabels(['']* len(df_plot['condition'].unique()), fontsize = 14, rotation = 45, fontweight='bold',c = 'black')
        
    #ax = fig.add_subplot(int(np.ceil(len(labels)/3)),3, i + 2)
    #ax.spines["top"].set_visible(False)
    #ax.spines["right"].set_visible(False)
    #ax.spines["left"].set_visible(False)
    #ax.spines["bottom"].set_visible(False)
    #ax.axis('off')
    #ax.legend([Line2D([0], [0], color=lst_color[0], lw=3), 
    #            Line2D([0], [0], color=lst_color[2], lw=3), 
    #            Line2D([0], [0], color=lst_color[1], lw=3)], 
    #          [lst_groups[0], 
    #           lst_groups[1],
    #           lst_groups[2]], 
    #          loc ='upper left', 
    #          facecolor='white',
    #          fontsize='x-large')
    fig.tight_layout() 
    return  fig, df_full

if False:
    lst_groups = list(df_diff_diff['Condition'].unique())
    df_data = df_diff_diff
    col_group="Condition"
    labels= variables
    boot = 'mean'  
    boot_size = 5000
    title=diff_diff_title 
    lst_color=['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet', 'deeppink', 'cyan', 'green']
    save_path = save_path
    fwr_correction = True

def plot_boxes_errorbar(lst_groups,df_data,col_group,labels, boot = 'median',  
               boot_size = 2000,title='', lst_color=None, save_path = None, fwr_correction = True, contrasts = False):
    fig = plt.figure(figsize=(18, 10), facecolor='white')   
    fig.suptitle(title, fontsize = 20, fontweight='bold')            
    
    for i, label in enumerate(labels):
        
        print(label)
        ax = fig.add_subplot(len(labels),1, i + 1)
        df_plot = df_data.dropna(subset =[label,col_group], axis = 0).copy()
            
        lst_ala = []
        lst_col = []
        lst_boo = []
           
        for col in lst_groups:   
            vals   = df_plot.loc[df_plot[col_group]==col,label].values
            
            lst_ala.append(vals)
            lst_col.append(col)
            if fwr_correction:   
                alpha = 1 - (0.05 / len(lst_groups))
            else: 
                alpha = 1 - (0.05)
            if boot == 'mean':
                dict_b = bootstrapping(vals,
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
                    c = 'grey')
        
        if contrasts:
            ax.axhline(0, c = 'black',ls='--')
        if 'Correlation' in label:
            ax.axhline(0.05, c = 'black',ls='--')
        ax.plot()
        ax.set_xticks(np.arange(0, len(lst_groups), 1))
        ax.set_xticklabels(['']*len(lst_groups),fontsize=14, fontweight='bold', rotation=45, ha='right', c = 'black' )#
        
        ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 14, fontweight='bold',c = 'black')
        ax.set_title(label, fontsize = 16, fontweight='bold', c = 'black' )
        df_boo.index = lst_groups
        df_boo.index.rename(label, inplace = True)
        if 'Face' in title:
            save = save_path + '/Bootstrapped_comparison_Face.csv'
        elif 'NBack' in title:
            save = save_path + '/Bootstrapped_comparison_NBack.csv'
        else:
            save = os.path.join(save_path, 'Bootstrapped_comparison_'+ label +'.csv')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df_boo.to_csv(save, index = True, header = (not os.path.exists(save)), mode = 'a', decimal=',', sep=';' )
    ax.set_xticklabels(lst_groups,fontsize=14, fontweight='bold', rotation=45, ha='right', c = 'black' )# 
    fig.tight_layout()

    return df_boo

def plot_eva_audio(lst_groups,df_data,col_group, col_ax,  labels, boot = 'median',  
               boot_size = 2000,title='', lst_color=None, save_path = None, fwr_correction = True):
    fig = plt.figure(figsize=(18, 10), facecolor='white')   
    fig.suptitle(title, fontsize = 20, fontweight='bold')            
    
    for i, label in enumerate(labels):
        
        print(label)
        ax = fig.add_subplot(4,int(4), i + 1)
        
        df_plot = df_data.dropna(subset =[col_group], axis = 0).copy()
        df_plot = df_plot.loc[df_plot[col_ax]== label]    
        lst_ala = []
        lst_col = []
        lst_boo = []
           
        for col in lst_groups:   
            vals   = df_plot.loc[df_plot[col_group]==col,'value'].values
            
            lst_ala.append(vals)
            lst_col.append(col)
            if fwr_correction:   
                alpha = 1 - (0.05 / len(lst_groups))
            else: 
                alpha = 1 - (0.05)
            if boot == 'mean':
                dict_b = bootstrapping(vals,
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
                    c = 'grey')
            
        ax.plot()
        ax.set_xticks(np.arange(0, len(lst_groups), 1))
        
        #if i == len(label):
        #    ax.set_xticklabels(lst_groups,fontsize=14, fontweight='bold', rotation=45, ha='right', c = 'black' )# 
        #else:     
        ax.set_xticklabels(len(lst_groups) * [''],fontsize=14, fontweight='bold', rotation=45, ha='right', c = 'black' )
        ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 14, fontweight='bold',c = 'black')
        ax.set_title(label, fontsize = 16, fontweight='bold', c = 'black' )
        df_boo.index = lst_groups
        df_boo.index.rename(label, inplace = True)
        
        if 'Correlation' in label:
            ax.axhline(0.05, c = 'black',ls='--')
        if 'positive' in label: 
            try:
                ax.axhline(df_boo.loc['prob_pos','lower'], c = 'blue',ls='--')
            except: 
                try:
                    ax.axhline(df_boo.loc['Prob Pos','lower'], c = 'blue',ls='--')
                except: 
                    ax.axhline(df_boo.loc['positive','lower'], c = 'blue',ls='--')
        elif 'negative' in label:
            try:
                ax.axhline(df_boo.loc['prob_neg','lower'], c = 'blue',ls='--')
            except: 
                try:
                    ax.axhline(df_boo.loc['Prob Neg','lower'], c = 'blue',ls='--')
                except: 
                    ax.axhline(df_boo.loc['negative','lower'], c = 'blue',ls='--')       
                    
        elif 'neutral' in label:        
            try:
                ax.axhline(df_boo.loc['prob_neu','lower'], c = 'blue',ls='--')
            except: 
                try:
                    ax.axhline(df_boo.loc['Prob Neu','lower'], c = 'blue',ls='--')
                except: 
                    ax.axhline(df_boo.loc['neutral','lower'], c = 'blue',ls='--')               
        if 'Face' in title:
            save = save_path + '/Bootstrapped_comparison_Face.csv'
        elif 'NBack' in title:
            save = save_path + '/Bootstrapped_comparison_NBack.csv'
        else:
            save = os.path.join(save_path, 'Bootstrapped_comparison_'+ label +'.csv')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df_boo.to_csv(save, index = True, header = (not os.path.exists(save)), mode = 'a', decimal=',', sep=';' )
    fig.tight_layout()

    return df_boo

# =============================================================================
# ENCODING 
# =============================================================================

def plot_boxes(lst_groups,df_data,col_group,label, boot = 'median',  
               boot_size = 5000,title='', ax = None,lst_color=None, alpha = 0.95):
    
    lst_ala = []
    lst_col = []
    lst_boo = []
    
    for col in lst_groups:    
        vals   = df_data.loc[df_data[col_group]==col,label].values
        
        lst_ala.append(vals)
        lst_col.append(col)
        
        if boot == 'mean':
            dict_b = bootstrapping(vals,
                     numb_iterations = boot_size,
                     alpha = alpha,
                     as_dict = True,
                     func = 'mean')
            lst_boo.append(dict_b)
        
    if boot == 'mean':
        df_boo = pd.DataFrame(lst_boo)
        m      = df_data[label].mean()
    else:
        m      = df_data[label].median()
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(16, 11))
    
    if boot == 'mean':
        bplot = ax.boxplot(lst_ala,
                   notch      = True,
                   usermedians= df_boo['mean'].values,
                   conf_intervals=df_boo.loc[:,['lower','upper']].values,
                   showfliers = False,
                   whis       = [5,95],
                   labels     = lst_col)
    else:
        bplot = ax.boxplot(lst_ala,
               notch      = True,
               bootstrap  = boot_size,
               showfliers = False,
               whis       = [5,95],
               labels     = lst_col)
    
    if lst_color is not None:
        for box, color in zip(bplot['boxes'], lst_color):
            box.set(color=color)
        
    ax.plot()
    ax.set_title(title  + ' 95% CI of '+ boot ,fontsize = 20)
    ax.set_xlabel('Conditions',fontsize = 18)    
    if False: fig.legend([Line2D([0], [0], color=lst_color[0], lw=3), 
               Line2D([0], [0], color=lst_color[1], lw=3), 
               Line2D([0], [0], color='blue', lw=3), 
               Line2D([0], [0], color='orange', lw=3)], 
              [lst_groups[0], 
               lst_groups[1],

               'Bootstrapped Mean over all Conditions', 
               'Bootstrapped Mean within Condition'], 
              loc = 'lower left',
              facecolor='white',
              fontsize='x-large')
    fig.tight_layout() 
    return df_boo
