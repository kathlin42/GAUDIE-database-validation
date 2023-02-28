# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 22:41:17 2021

@author: hirning
"""

# =============================================================================
# BIDS CONVERSION OF THE AUDIO SEQUENCES
# =============================================================================

# =============================================================================
# Import Packages
# =============================================================================

import os
import shutil
import pandas as pd 
import numpy as np
import mne_bids
from mne_bids import write_raw_bids, BIDSPath, print_dir_tree
import config_analysis
# =============================================================================
# Creating Directories
# =============================================================================

bids_root = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
source_data = os.path.join(bids_root, 'source_data', 'behavioral')
questionnaire_data = os.path.join(bids_root, 'source_data', 'questionnaires')
task = 'gaudieaudiovalidation'
datatype = 'beh'
questionnaire = 'demographics_questionnaires'
if not os.path.exists(bids_root):
    os.makedirs(bids_root)  
if not os.path.exists(os.path.join(bids_root, 'stimulus_material')):
    os.makedirs(os.path.join(bids_root, 'stimulus_material'))    
if not os.path.exists(os.path.join(bids_root, questionnaire)):
    os.makedirs(os.path.join(bids_root, questionnaire))    
if not os.path.exists(os.path.join(bids_root, 'raw_data')):
    os.makedirs(os.path.join(bids_root, 'raw_data')) 
# =============================================================================
# Write BIDS Dataset Description
# =============================================================================
mne_bids.make_dataset_description(path = bids_root, name = 'gaudie_audio_validation',  authors=['Katharina Lingelbach', 'Mathias Vukelic', 'Jochem Rieger'], 
                                  dataset_type='raw', data_license = 'Open Data Commons license', overwrite=True, verbose=None)

# =============================================================================
# Loop over Participants
# =============================================================================
subj_list = os.listdir(source_data)
subj_list = [sub for sub in subj_list if len(sub) == 3]
subj_list_bids =  ['00' + str(int(sub) + 1) if (int(sub) + 1) < 10 else '0' + str(int(sub) + 1) for sub in subj_list]

for isubj, subj in enumerate(subj_list):
    subject_id = subj_list_bids[isubj]
    subj_save = os.path.join(bids_root, 'raw_data', 'sub-' + subject_id, datatype)
    if not os.path.exists(subj_save):
        os.makedirs(subj_save)    
    valence_ratings = [file for file in os.listdir(os.path.join(source_data, subj)) if 'Valence' in file]
    arousal_ratings = [file for file in os.listdir(os.path.join(source_data, subj)) if 'Arousal' in file]    
    other_ratings = [file for file in os.listdir(os.path.join(source_data, subj)) if 'ratings' in file]
    # =============================================================================
    # Loop over Ratings and provide meaningful names
    # =============================================================================
    for file in arousal_ratings:           
        df = pd.read_csv(os.path.join(source_data, subj, file), decimal='.', sep=';')
        df.columns = ['playback_time', 'slider_position']
        if 'positive_funny' in file: 
            label = 'positive'
        elif 'negative' in file: 
            label = 'negative'
        elif 'neutral' in file: 
            label = 'neutral'
        ID = int(file[-18:-16]) + 1
        if ID < 10:
            ID = '0' + str(ID)
        else: 
            ID = str(ID)
            
        df.to_csv(os.path.join(subj_save,'sub-' + subject_id + '_task-' + task + '_SAM_arousal' +'_' + label + '_' + ID + '.tsv'), sep='\t', decimal='.',  header=True, index = False)
    for file in valence_ratings:           
        df = pd.read_csv(os.path.join(source_data, subj, file), decimal='.', sep=';')
        df.columns = ['playback_time', 'slider_position']
        if 'positive_funny' in file: 
            label = 'positive'
        elif 'negative' in file: 
            label = 'negative'
        elif 'neutral' in file: 
            label = 'neutral'
        ID = int(file[-18:-16]) + 1
        if ID < 10:
            ID = '0' + str(ID)
        else: 
            ID = str(ID)
        df.to_csv(os.path.join(subj_save,'sub-' + subject_id + '_task-' + task + '_SAM_valence' + '_' + label + '_' + ID + '.tsv'), sep='\t', decimal='.', header=True, index = False)

    for file in other_ratings:           
        df = pd.read_csv(os.path.join(source_data, subj, file), decimal='.', sep=';', header = 0, encoding = 'ISO-8859-1')
        df = df.loc[(df['audio'] != 'audio') & (df['audio'] != 'Testaudio.wav') &(df['audio'] !='normalized_-20dB_test_00.wav')]
        df = df.reset_index(drop = True)
        lst_condition = []
        list_id = []
        list_audio = []
        for index, row in df.iterrows():
            if 'negative' in row['audio']:  
                lst_condition.append('negative')
                list_id.append(row['audio'][26:28])
                list_audio.append('negative' + '_' + row['audio'][26:28])
            elif 'positive' in row['audio']: 
                lst_condition.append('positive')
                list_id.append(row['audio'][32:34])
                list_audio.append('positive' + '_' + row['audio'][32:34])
            elif'neutral' in row['audio']:
                lst_condition.append('neutral')
                list_id.append(row['audio'][25:27])
                list_audio.append('neutral' + '_' + row['audio'][25:27])
        df['audio'] = list_audio
        df['condition'] = lst_condition
        list_emo = ['Angst', 'Ekel', 'Trauer', 'Wut', 'Überraschung', 'Freude']
        list_emo_new = ['Fear', 'Disgust', 'Sadness', 'Anger', 'Surprise', 'Joy']
        list_gew = ['Angst', 'Belustigung', 'Bereuen', 'Ekel', 'Entäuschung', 'Erleichterung', 'Freude', 'Hass', 'Interesse', 'Mitgefühl',
                    'Scham', 'Schuld', 'Stolz', 'Trauer', 'Verachtung', 'Vergnügen', 'Wut','Zufriedenheit']
        list_gew_new = ['Fear', 'Amusement', 'Regret', 'Disgust', 'Disappointment', 'Relief', 'Joy', 'Hatred', 'Interest', 'Compassion', 'Shame', 'Guilt',
                      'Pride', 'Sadness', 'Contempt', 'Pleasure', 'Anger', 'Contentment']
        df['emotion'] = df['emotion'].replace(dict(zip(list_emo, list_emo_new)))
        df['geneva_wheel'] = df['geneva_wheel'].replace(dict(zip(list_gew, list_gew_new)))
        df.columns = ['audio', 'geneva_wheel_emotion', 'geneva_wheel_strength', 'basic_emotion', 'familiarity','dominance','condition']

        df.to_csv(os.path.join(subj_save,'sub-' + subject_id + '_task-' + task + '_postpresentation_ratings.tsv'), sep='\t', decimal='.', header=True, index = False)
# =============================================================================
# Loop over stimuli and provide meaningful names
# =============================================================================
for iaudio, audio in enumerate(os.listdir(os.path.join(source_data,  'audio'))):
    if 'positive_funny' in audio: 
        label = 'positive'
    elif 'negative' in audio: 
        label = 'negative'
    elif 'neutral' in audio: 
        label = 'neutral'
    ID = int(audio[-6:-4]) + 1  
    if ID < 10:
        ID = '0' + str(ID)
    else: 
        ID = str(ID)
    new_file_name = label + '_' + ID + '.wav'  
    shutil.copy(os.path.join(source_data,  'audio', audio), os.path.join(bids_root, 'stimulus_material', new_file_name))
    
df = pd.read_csv(os.path.join(questionnaire_data, 'Validierungsstudie_Screening.csv'), decimal='.', sep=';', header = 0, encoding = 'ISO-8859-1')
df['ID'] = df['ID'] + 1
df.to_csv(os.path.join(bids_root, questionnaire,'demographics_and_screening.tsv'), sep='\t', decimal='.', header=True, index = False)

df = pd.read_csv(os.path.join(questionnaire_data, 'Validierungsstudie_Questionnaires.csv'), decimal='.', sep=';', header = 0, encoding = 'ISO-8859-1')
df['Bitte bitten Sie die Versuchsleiter*in Ihre ID einzugeben.'] = df['Bitte bitten Sie die Versuchsleiter*in Ihre ID einzugeben.'] + 1
df = df.rename(columns = {'Bitte bitten Sie die Versuchsleiter*in Ihre ID einzugeben.': 'ID'})
df.to_csv(os.path.join(bids_root, questionnaire,'questionnaire_data.tsv'), sep='\t', decimal='.', header=True, index = False)

