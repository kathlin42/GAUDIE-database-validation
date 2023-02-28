# =============================================================================
# Directories and Imports
# =============================================================================
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pingouin as pg
import helper_plotting as pb
import config_analysis

repository = os.path.join(config_analysis.project_root, "gaudie_audio_validation")

data_path = os.path.join(repository, 'derivatives', 'questionnaires')
mpl.rc('font',family='Times New Roman', weight = 'bold')

# =============================================================================
# Prepare data
# =============================================================================
df_scores = pd.read_csv(os.path.join(data_path, 'demographics_scores.csv'), sep=';', decimal=',')
df_scores['ID'] = [ int(i) + 1 for i in df_scores['ID']]
# =============================================================================
# save
# =============================================================================
save_path = data_path
# =============================================================================
# Plotting Param
# =============================================================================
lst_color=['#003366', '#4DC3FF', '#00FFD5', '#00CC88', '#006655']

# =============================================================================
# BFI-K
# =============================================================================
dict_bfik_ref = {'extraversion' : (3.48 + 3.46) / 2, 
                 'agreeableness' : (3.02 + 3.49) / 2,
                 'conscientiousness' : (3.53 + 3.46) / 2,
                 'neuroticism' : (2.88 + 2.95) / 2,
                 'openness' : (3.96 + 3.69) / 2}
titles = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
fig, axes = plt.subplots(len(['extraversion', 'agreeableness', 'conscientiousness',
       'neuroticism', 'openness']),1, figsize=(10,12),sharex=True, sharey=True)
axes = axes.flatten()    
for i, (scale, ax) in enumerate(zip(['extraversion', 'agreeableness', 'conscientiousness',
       'neuroticism', 'openness'], axes)):

    ax.bar(df_scores['ID'].astype(str), df_scores[scale], color = lst_color[i], edgecolor = "black")#)
    if ax == axes[-1]:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([i + 1 for i in ax.get_xticks()], fontsize = 16,fontweight='bold',c = 'black')
        ax.set_xlabel('Subject IDs', fontsize = 20,fontweight='bold',c = 'black')
    
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(), fontsize = 16,fontweight='bold',c = 'black')
    ax.set_title(titles[i], fontsize = 22, fontweight='bold')  
    ax.axhline(dict_bfik_ref[scale], color = 'blue', linestyle='--')  
    
    
fig.suptitle('Big Five - German Short Version (21 Items)',  fontsize = 25, fontweight='bold') 
fig.text(0.01, 0.5, 'Scale (1 - 5)', va='center', rotation='vertical', fontsize = 22,fontweight='bold',c = 'black')
# set the spacing between subplots
fig.tight_layout(rect = [0.025, 0, 1, 0.98],h_pad=2.5)          
# save the file 
fig.savefig(os.path.join(save_path, 'Big_Five_per_sub.svg'),bbox_inches='tight')
plt.close()

for i_scale, scale_plot in enumerate(['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']):
    if i_scale == 0:
        df_plot = pd.DataFrame()
        df_plot['ID'] = df_scores['ID']
        df_plot['scale'] = scale_plot
        df_plot['BIG 5 - Short Version'] = df_scores.loc[: , scale_plot]
    else: 
        df_temp = pd.DataFrame()
        df_temp['ID'] = df_scores['ID']
        df_temp['scale'] = scale_plot
        df_temp['BIG 5 - Short Version'] = df_scores.loc[: , scale_plot]
        df_plot = df_plot.append(df_temp)
        
        
fig, df_boot = pb.plot_boxplots_plus_ref(sorted(list(df_plot['scale'].unique())),
                 df_plot,
                 'scale',
                 ['BIG 5 - Short Version'], 
                 boot = 'mean', 
                 boot_size = 5000,
                 title='Big Five Short Version', 
                 lst_color=lst_color, 
                 save_path = save_path, 
                 refs = dict_bfik_ref, 
                 fwr_correction = False)


fig.savefig(os.path.join(save_path, 'Bootstrapped_Big_Five.svg'))
plt.close()
df_boot.to_csv(os.path.join(save_path, 'Bootstrapped_Big_Five.csv'), index = True, header = True, decimal=',', sep=';' )


lst_color=['#A9A9A9', '#004D33', '#008040', '#00CC66', '#80FF95']

# =============================================================================
# Depression
# =============================================================================
fig, axes = plt.subplots(1,1, figsize=(10,6),sharex=True, sharey=True)
axes.bar(df_scores['ID'].astype(str), df_scores['depression_mean'], color = lst_color[0], edgecolor = 'black')
axes.set_xticks(ax.get_xticks())
axes.set_xticklabels([i + 1 for i in ax.get_xticks()], fontsize = 16,fontweight='bold',c = 'black')
axes.set_xlabel('Subject IDs', fontsize = 20,fontweight='bold',c = 'black')
axes.set_yticks(np.linspace(0, 1,6))
axes.set_yticklabels(list(np.round(np.linspace(0, 1,6),1)), fontsize = 16,fontweight='bold',c = 'black')
axes.set_ylabel('Depression Scale', fontsize = 20,fontweight='bold',c = 'black') 
axes.axhline(0.5, color = 'blue', linestyle='--')
fig.suptitle('2-Items Depression Scale',  fontsize = 25, fontweight='bold') 
# set the spacing between subplots
fig.tight_layout()     
# save the file 
fig.savefig(os.path.join(save_path, 'Depression_per_sub.svg'),bbox_inches='tight')
plt.close()

# =============================================================================
# WHO-5
# =============================================================================
fig, axes = plt.subplots(1,1, figsize=(10,6),sharex=True, sharey=True)
axes.bar(df_scores['ID'].astype(str), df_scores['who5_sum'], color = lst_color[1], edgecolor = 'black')

axes.set_xticks(ax.get_xticks())
axes.set_xticklabels([i + 1 for i in ax.get_xticks()], fontsize = 16,fontweight='bold', c = 'black')
axes.set_xlabel('Subject IDs', fontsize = 20,fontweight='bold',c = 'black')
axes.set_yticks(np.linspace(0, 100,6))
axes.set_yticklabels(list(np.linspace(0, 100,6)), fontsize = 16,fontweight='bold',c = 'black')
axes.set_ylabel('WHO-5 Scale', fontsize = 20,fontweight='bold',c = 'black') 
axes.axhline(50, color = 'blue', linestyle='--')
fig.suptitle('Subjective Wellbeing - WHO-5 Scale',  fontsize = 25, fontweight='bold') 
# set the spacing between subplots
fig.tight_layout()        
# save the file 
fig.savefig(os.path.join(save_path, 'WHO-5_per_sub.svg'),bbox_inches='tight')
plt.close()

# =============================================================================
# SWLS
# =============================================================================
fig, axes = plt.subplots(1,1, figsize=(10,6),sharex=True, sharey=True)
axes.bar(df_scores['ID'].astype(str), df_scores['swls_sum'], color = lst_color[2], edgecolor = 'black')
axes.set_xticks(ax.get_xticks())
axes.set_xticklabels([i + 1 for i in ax.get_xticks()], fontsize = 16,fontweight='bold',c = 'black')
axes.set_xlabel('Subject IDs', fontsize = 20,fontweight='bold',c = 'black')
axes.set_yticks(np.linspace(0, 35,8))
axes.set_yticklabels(list(np.linspace(0, 35,8)), fontsize = 16,fontweight='bold',c = 'black')
axes.set_ylabel('SWL Scale', fontsize = 20,fontweight='bold',c = 'black')
axes.axhline(20, color = 'blue', linestyle='--')
fig.suptitle('Statisfaction with Life Scale',  fontsize = 25, fontweight='bold') 
# set the spacing between subplots
fig.tight_layout()        
# save the file 
fig.savefig(os.path.join(save_path, 'SWL_per_sub.svg'),bbox_inches='tight')
plt.close()

# =============================================================================
# STAI-S
# =============================================================================
fig, axes = plt.subplots(1,1, figsize=(10,6),sharex=True, sharey=True)
axes.bar(df_scores['ID'].astype(str), df_scores['state_anxiety_sum'], color = lst_color[3], edgecolor = 'black')
axes.set_xticks(ax.get_xticks())
axes.set_xticklabels([i + 1 for i in ax.get_xticks()], fontsize = 16,fontweight='bold',c = 'black')
axes.set_xlabel('Subject IDs', fontsize = 20,fontweight='bold',c = 'black')
axes.set_yticks(np.linspace(0, 80,9))
axes.set_yticklabels(list(np.linspace(0, 80,9)), fontsize = 16,fontweight='bold',c = 'black')
axes.set_ylabel('STAI-S Scale', fontsize = 20,fontweight='bold',c = 'black') 
axes.axhline(40, color = 'blue', linestyle='--')
fig.suptitle('State Anxiety Index',  fontsize = 25, fontweight='bold') 
# set the spacing between subplots
fig.tight_layout()        
# save the file 
fig.savefig(os.path.join(save_path, 'STAIS_per_sub.svg'),bbox_inches='tight')
plt.close()

# =============================================================================
# STAI-T
# =============================================================================

fig, axes = plt.subplots(1,1, figsize=(10,6),sharex=True, sharey=True)
axes.bar(df_scores['ID'].astype(str), df_scores['trait_anxiety_sum'], color = lst_color[4], edgecolor = 'black')
axes.set_xticks(ax.get_xticks())
axes.set_xticklabels([i + 1 for i in ax.get_xticks()], fontsize = 16,fontweight='bold',c = 'black')
axes.set_xlabel('Subject IDs', fontsize = 20,fontweight='bold',c = 'black')
axes.set_yticks(np.linspace(0, 80,9))
axes.set_yticklabels(list(np.linspace(0, 80,9)), fontsize = 16,fontweight='bold',c = 'black')
axes.set_ylabel('STAI-T Scale', fontsize = 20,fontweight='bold',c = 'black') 
axes.axhline(40, color = 'blue', linestyle='--')
fig.suptitle('Trait Anxiety Index',  fontsize = 25, fontweight='bold') 
# set the spacing between subplots
fig.tight_layout()        
# save the file 
fig.savefig(os.path.join(save_path, 'STAIT_per_sub.svg'))
plt.close()

# =============================================================================
# Bootstrapped Scales
# =============================================================================
list_norm_ref = []
dict_norm = {}
dict_ref = {'depression_mean': 0.5, 
            'who5_sum': 50, 
            'swls_sum': 20, 
            'state_anxiety_sum': 40, 
            'trait_anxiety_sum': 40}

names = ['Depression','WHO-5', 'SWLS', 'STAIS', 'STAIT']
for i_scale, scale_plot in enumerate(['depression_mean','who5_sum', 'swls_sum', 'state_anxiety_sum', 'trait_anxiety_sum']):
    if i_scale == 0:
        df_plot = pd.DataFrame()
        df_plot['ID'] = df_scores['ID']
        df_plot['scale'] = names[i_scale]
        data = df_scores.loc[: , scale_plot] 
        df_plot['Well-Being Scales'] = (data - data.min())/ (data.max() - data.min())
        list_norm_ref.append((dict_ref[scale_plot] - data.min())/ (data.max() - data.min()))
        dict_norm[names[i_scale]] = [data.min(), data.max()]
    else: 
        df_temp = pd.DataFrame()
        df_temp['ID'] = df_scores['ID']
        df_temp['scale'] =  names[i_scale]
        data = df_scores.loc[: , scale_plot] 
        df_temp['Well-Being Scales'] = (data - data.min())/ (data.max() - data.min())
        df_plot = df_plot.append(df_temp)
        list_norm_ref.append((dict_ref[scale_plot] - data.min())/ (data.max() - data.min()))
        dict_norm[names[i_scale]] = [data.min(), data.max()]
fig, df_boot = pb.plot_boxplots_plus_ref(['Depression','WHO-5', 'SWLS', 'STAIS', 'STAIT'],
                 df_plot,
                 'scale',
                 ['Well-Being Scales'], 
                 boot = 'mean', 
                 boot_size = 5000,
                 title='Well-Being Scales', 
                 lst_color=lst_color, 
                 save_path = save_path, 
                 refs = {'Depression': list_norm_ref[0], 'WHO-5': list_norm_ref[1], 'SWLS':  list_norm_ref[2], 'STAIS':  list_norm_ref[3], 'STAIT':  list_norm_ref[4]}, 
                 fwr_correction = False)

for  i_scale, scale_plot in enumerate(['Depression','WHO-5', 'SWLS', 'STAIS', 'STAIT']):
    df_boot.loc[df_boot.index == scale_plot, 'lower'] = (df_boot.loc[df_boot.index == scale_plot, 'lower'].values * (dict_norm[scale_plot][1] - dict_norm[scale_plot][0])) +  dict_norm[scale_plot][0]
    df_boot.loc[df_boot.index == scale_plot, 'mean'] = (df_boot.loc[df_boot.index == scale_plot, 'mean'].values * (dict_norm[scale_plot][1] - dict_norm[scale_plot][0])) +  dict_norm[scale_plot][0]
    df_boot.loc[df_boot.index == scale_plot, 'upper'] = (df_boot.loc[df_boot.index == scale_plot, 'upper'].values * (dict_norm[scale_plot][1] - dict_norm[scale_plot][0])) +  dict_norm[scale_plot][0]

fig.savefig(os.path.join(save_path, 'Bootstrapped_Well-Being.svg'))
df_boot.to_csv(os.path.join(save_path, 'Bootstrapped_WellBeing.csv'), index = True, header = True, decimal=',', sep=';' )
plt.close()


# =============================================================================
#  Gender
# =============================================================================

fig, axes = plt.subplots(1,1, figsize=(8,6),sharex=True, sharey=True)
pie_data = []
pie_labels = []
for i, value in enumerate(df_scores['gender'].unique()):
    pie_data.append(df_scores['gender'].value_counts()[i] / len(df_scores['gender']) * 100)
    pie_labels.append(df_scores['gender'].value_counts().index[i])
axes.pie(pie_data, labels=pie_labels, colors=lst_color[-len(pie_labels):], autopct='%1.1f%%')
fig.suptitle('Gender Distribution',  fontsize = 25, fontweight='bold') 
# set the spacing between subplots
fig.tight_layout()        
# save the file 
fig.savefig(os.path.join(save_path, 'Gender_pie.svg'))
plt.close()

# =============================================================================
# Age
# =============================================================================

fig, axes = plt.subplots(1,1, figsize=(8,6),sharex=True, sharey=True)
axes.hist(df_scores['age'], color = lst_color[3], edgecolor = 'black')
axes.set(xlabel= 'Age', ylabel= 'Frequency')  
axes.axvline(df_scores['age'].mean(), color = 'blue', linestyle='--')
axes.text(x=df_scores['age'].mean() + 0.01, y=axes.get_ylim()[0] + 1, s='Mean', fontsize=10, color = 'blue')
axes.axvline(df_scores['age'].median(), color = 'orange', linestyle='--')
axes.text(x=df_scores['age'].median() + 0.01, y=axes.get_ylim()[0] + 1, s='Median', fontsize=10, color = 'orange')
fig.suptitle('Age Distribution',  fontsize = 25, fontweight='bold') 
# set the spacing between subplots
fig.tight_layout()        
# save the file 
fig.savefig(os.path.join(save_path, 'Age_hist.svg'))
plt.close()

# =============================================================================
# Education
# =============================================================================

fig, axes = plt.subplots(1,1, figsize=(8,6),sharex=True, sharey=True)
bar_data = []
bar_labels = []
for i, value in enumerate(df_scores['education'].unique()):
    bar_data.append(df_scores['education'].value_counts()[i] / len(df_scores['education']) * 100)
    bar_labels.append(df_scores['education'].value_counts().index[i])
axes.bar(bar_labels, bar_data, color = lst_color[3], edgecolor = 'black')
for p in axes.patches:
    axes.annotate(str(np.round(p.get_height(),2)) + ' %', (p.get_x() + 0.3, p.get_height() + 0.5))
axes.set(xlabel= 'education', ylabel= 'Frequency')  
fig.suptitle('Education Distribution',  fontsize = 25, fontweight='bold') 
# set the spacing between subplots
fig.tight_layout()        
# save the file 
fig.savefig(os.path.join(save_path, 'Education_hist.svg'))
plt.close()

# =============================================================================
# Profession
# =============================================================================

fig, axes = plt.subplots(1,1, figsize=(8,6),sharex=True, sharey=True)
bar_data = []
bar_labels = []
for i, value in enumerate(df_scores['profession'].unique()):
    bar_data.append(df_scores['profession'].value_counts()[i] / len(df_scores['profession']) * 100)
    bar_labels.append(df_scores['profession'].value_counts().index[i])
axes.bar(bar_labels, bar_data, color = lst_color[1], edgecolor = 'black')
for p in axes.patches:
    axes.annotate(str(np.round(p.get_height(),2)) + ' %', (p.get_x() + 0.2, p.get_height() + 0.5))
axes.set(xlabel= 'profession', ylabel= 'Frequency')  
fig.suptitle('Profession Distribution',  fontsize = 25, fontweight='bold') 
# set the spacing between subplots
fig.tight_layout()        
# save the file 
fig.savefig(os.path.join(save_path, 'Profession_hist.svg'))
plt.close()