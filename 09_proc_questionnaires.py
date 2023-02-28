# =============================================================================
# Directories and Imports
# =============================================================================
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
import config_analysis

bids_root = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
source_data = os.path.join(bids_root, 'source_data', 'questionnaires')

# =============================================================================
# Prepare data
# =============================================================================

df_screening = pd.read_csv(source_data + 'Validierungsstudie_Screening.csv', sep=';', decimal=',', encoding = 'ISO-8859-1')
df_questions = pd.read_csv(source_data + 'Validierungsstudie_Questionnaires.csv', sep=';', decimal=',', encoding = 'ISO-8859-1')

# =============================================================================
# save
# =============================================================================

save_path = os.path.join(bids_root, 'derivatives', 'questionnaires')
if not os.path.exists("{}".format(save_path)):
    print('creating path for saving')
    os.makedirs("{}".format(save_path))
# =============================================================================
# helper
# =============================================================================
def sublist(cols, idx_list):
    for i_idx, idx in enumerate(idx_list):
        if i_idx == 0: 
            sublist = [cols[idx]]
        else: 
            sublist.append(cols[idx])
    return sublist
# =============================================================================
# ID
# =============================================================================
df_scores = pd.DataFrame()
df_alpha = pd.DataFrame(columns = ['score', 'cronbach_a', '5thCI', '95thCI'])
df_scores['ID'] = df_questions.iloc[: , 4]

# =============================================================================
# BFI-K
# =============================================================================

'''
Cue for BFI-K - 'Ich...'
'''
bfik_cols = [col for col in df_questions.columns if 'Ich...' in col]
# Reverse
bfik_cols_r = sublist(bfik_cols, [0, 1, 7, 8, 10, 11, 16, 20])
# Extraversion
bfik_cols_e = sublist(bfik_cols, [0, 5, 10, 15])
# Aggreeableness
bfik_cols_a = sublist(bfik_cols, [1, 6, 11, 16])
# Consciousness
bfik_cols_c = sublist(bfik_cols, [2, 7, 12, 17])
# Neuroticism
bfik_cols_n = sublist(bfik_cols, [3, 8, 13, 18])     
# Openness
bfik_cols_o = sublist(bfik_cols, [4, 9, 14, 19, 20])

# Encoding in scores
df_questions[bfik_cols] = df_questions[bfik_cols].replace({
    'sehr unzutreffend': 1,
    'eher unzutreffend': 2,
    'weder noch': 3,
    'eher zutreffend': 4,
    'sehr zutreffend': 5})
# Reverse coded items
df_questions[bfik_cols_r] = df_questions[bfik_cols_r].replace({
    1: 5,
    2: 4,
    4: 2,
    5: 1})


# Computation of mean scores
df_scores['extraversion'] = df_questions[bfik_cols_e].mean(axis=1)
df_scores['agreeableness'] = df_questions[bfik_cols_a].mean(axis=1)
df_scores['conscientiousness'] = df_questions[bfik_cols_c].mean(axis=1)
df_scores['neuroticism'] = df_questions[bfik_cols_n].mean(axis=1)
df_scores['openness'] = df_questions[bfik_cols_o].mean(axis=1)

alpha_e = pg.cronbach_alpha(data=df_questions[bfik_cols_e])
alpha_a = pg.cronbach_alpha(data=df_questions[bfik_cols_a])
alpha_c = pg.cronbach_alpha(data=df_questions[bfik_cols_c])
alpha_n = pg.cronbach_alpha(data=df_questions[bfik_cols_n])
alpha_o = pg.cronbach_alpha(data=df_questions[bfik_cols_o])

df_alpha.loc[len(df_alpha.index)] = ['bfik_e', alpha_e[0], alpha_e[1][0], alpha_e[1][1]]
df_alpha.loc[len(df_alpha.index)] = ['bfik_a', alpha_a[0], alpha_a[1][0], alpha_a[1][1]]
df_alpha.loc[len(df_alpha.index)] = ['bfik_c', alpha_c[0], alpha_c[1][0], alpha_c[1][1]]
df_alpha.loc[len(df_alpha.index)] = ['bfik_n', alpha_n[0], alpha_n[1][0], alpha_n[1][1]]
df_alpha.loc[len(df_alpha.index)] = ['bfik_o', alpha_o[0], alpha_o[1][0], alpha_o[1][1]]

# =============================================================================
# Depression
# =============================================================================
item_1 = 'Fühlten Sie sich im letzten Monat häufig niedergeschlagen, traurig bedrückt oder hoffnungslos?\xa0'
item_2 = 'Hatten Sie im letzten Monat deutlich weniger Lust und Freude an Dingen, die Sie sonst gerne tun?\xa0'
df_questions[[item_1,item_2]]
df_questions[[item_1,item_2]] = df_questions[[item_1,item_2]].replace({
                        'Nein': 0,
                        'Ja': 1})

# Computation of mean scores
df_scores['depression_sum'] = df_questions[[item_1,item_2]].sum(axis=1)
df_scores['depression_mean'] = df_questions[[item_1,item_2]].mean(axis=1)

alpha_d = pg.cronbach_alpha(data=df_questions[[item_1,item_2]])
df_alpha.loc[len(df_alpha.index)] = ['depression', alpha_d[0], alpha_d[1][0], alpha_d[1][1]]
# =============================================================================
# WHO-5
# =============================================================================

'''
Cue for WHO-5 - 'Wohlbefinden in den letzten'
'''
who5_cols = [col for col in df_questions.columns if 'Wohlbefinden in den letzten' in col]
df_questions[who5_cols] = df_questions[who5_cols].replace({
                        'Die ganze Zeit': 5,
                        'Meistens': 4,
                        'Etwas mehr als die Hälfte der Zeit': 3,
                        'Etwas weniger als die Hälfte der Zeit': 2,
                        'Ab und zu': 1,
                        'Zu keinem Zeitpunkt': 0})
# Computation of sum scores
df_scores['who5_sum'] = df_questions[who5_cols].sum(axis=1) * 4

alpha_who5 = pg.cronbach_alpha(data=df_questions[who5_cols])
df_alpha.loc[len(df_alpha.index)] = ['who5', alpha_who5[0], alpha_who5[1][0], alpha_who5[1][1]]
# =============================================================================
# Satisfaction with Life Scale 
# =============================================================================
'''
Cue for SWLS - 'Es folgen f�nf Aussagen, denen Sie zustimmen bzw. die Sie ablehnen k�nnen'

'''
swls_cols = [col for col in df_questions.columns if 'Es folgen fünf Aussagen, denen Sie zustimmen bzw. die Sie ablehnen können' in col]
df_questions[swls_cols] = df_questions[swls_cols].replace({
                        'trifft überhaupt nicht zu': 1,                    
                        'trifft nicht zu': 2,
                        'trifft eher nicht zu': 3,
                        'teils/teils': 4,
                        'trifft zu': 5,
                        'trifft eher zu': 6,
                        'trifft vollständig zu': 7})
# Computation of sum scores
df_scores['swls_sum'] = df_questions[swls_cols].sum(axis=1)

alpha_swls = pg.cronbach_alpha(data=df_questions[swls_cols])
df_alpha.loc[len(df_alpha.index)] = ['swls', alpha_swls[0], alpha_swls[1][0], alpha_swls[1][1]]

'''
- 31 - 35 Extremely satisfied  
- 26 - 30 Satisfied  
- 21 - 25 Slightly satisfied  
- 20      Neutral  
- 15 - 19 Slightly dissatisfied  
- 10 - 14 Dissatisfied  
- 5 -  9  Extremely dissatisfied 
'''

# =============================================================================
# STAI
# =============================================================================

stai_state_cols = [col for col in df_questions.columns if 'STAI - S' in col]
stai_trait_cols = [col for col in df_questions.columns if 'STAI-T' in col]
stai_cols = stai_state_cols + stai_trait_cols
stai_cols_r_s = sublist(stai_state_cols, [0, 1, 4, 7, 9, 10, 14, 15, 18, 19])
stai_cols_r_t = sublist(stai_trait_cols, [0, 5, 6, 9, 12, 15, 18]) 
stai_cols_r = stai_cols_r_s + stai_cols_r_t
# Encoding in scores
df_questions[stai_state_cols] = df_questions[stai_state_cols].replace({
                                'überhaupt nicht': 1,
                                'ein wenig': 2,
                                'ziemlich': 3,
                                'sehr': 4})
df_questions[stai_trait_cols] = df_questions[stai_trait_cols].replace({
                        'fast nie': 1,
                        'manchmal': 2,
                        'oft': 3,
                        'immer': 4})
# Reverse coded items
df_questions[stai_cols_r] = df_questions[stai_cols_r].replace({
    1: 4,
    2: 3,
    3: 2,
    4: 1})
# to numeric
for x in stai_cols:
    df_questions[x] = pd.to_numeric(df_questions[x])
    
# Computation of mean scores
df_scores['state_anxiety_sum'] = df_questions[stai_state_cols].sum(axis=1)
df_scores['state_anxiety_mean'] = df_questions[stai_state_cols].mean(axis=1)
df_scores['trait_anxiety_sum'] = df_questions[stai_trait_cols].sum(axis=1)
df_scores['trait_anxiety_mean'] = df_questions[stai_trait_cols].mean(axis=1)

alpha_stai_state = pg.cronbach_alpha(data=df_questions[stai_state_cols])
alpha_stai_trait = pg.cronbach_alpha(data=df_questions[stai_trait_cols])
df_alpha.loc[len(df_alpha.index)] = ['stai_state', alpha_stai_state[0], alpha_stai_state[1][0], alpha_stai_state[1][1]]
df_alpha.loc[len(df_alpha.index)] = ['stai_trait', alpha_stai_trait[0], alpha_stai_trait[1][0], alpha_stai_trait[1][1]]

# =============================================================================
# Screening and Start - Preparation
# =============================================================================

df_screening = df_screening.iloc[:,:14]
df_screening[df_screening.columns[1]] = df_screening[df_screening.columns[1]].replace({
    'weiblich': 'female',
    'männlich': 'male'})

profession = []
for idx, row in df_screening.iterrows():
    individual = []
    if row['Welche Tätigkeit üben Sie derzeit aus?  [Schüler/in]'] == 'Ja':
        individual.append('Pupil')
    if row['Welche Tätigkeit üben Sie derzeit aus?  [Studierende(r)]'] == 'Ja':
        individual.append('Student')
    if row['Welche Tätigkeit üben Sie derzeit aus?  [Auszubildende(r)]'] == 'Ja':
        individual.append('Apprentice')
    if row['Welche Tätigkeit üben Sie derzeit aus?  [Angestellte(r)]'] == 'Ja':
        individual.append('Employee')
    if row['Welche Tätigkeit üben Sie derzeit aus?  [Selbständige(r)]'] == 'Ja':
        individual.append('Freelancer')
    if row['Welche Tätigkeit üben Sie derzeit aus?  [in Elternzeit]'] == 'Ja':
        individual.append('Parental time')
    if row['Welche Tätigkeit üben Sie derzeit aus?  [Arbeitslose(r)]'] == 'Ja':
        individual.append('Unemployed')
    if row['Welche Tätigkeit üben Sie derzeit aus?  [Rentner(in)]'] == 'Ja':
        individual.append('Pensioner')
    if len(individual) > 1:
        individual = str(individual[0]) + ' & ' + str(individual[1])
        profession.append(individual)
    else: 
        try:
            profession.append(individual[0])
        except:
            profession.append('Other')

cols_profession = [col for col in df_screening.columns if 'Welche Tätigkeit üben Sie derzeit aus?' in col]
df_screening = df_screening.drop(cols_profession, axis = 1)
df_screening['profession'] = profession
df_screening.columns = ['ID', 'gender', 'age', 'education', 'language', 'profession']
df_screening['ID'] = df_screening['ID'] - 1

# =============================================================================
# Merge and Save
# =============================================================================

df_scores = pd.merge(df_scores, df_screening, on = "ID", how = "inner")
df_scores.to_csv(os.path.join(save_path, 'demographics_scores.csv'), index=False, decimal=',', sep=';')
df_alpha.to_csv(os.path.join(save_path, 'cronbach_alpha.csv'), index=False, decimal=',', sep=';')