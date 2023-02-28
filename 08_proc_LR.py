# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:30:09 2021

@author: hirning
"""
# =============================================================================
# Import Packages
# =============================================================================

import os
import pandas as pd 
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import pickle
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import RepeatedKFold, cross_val_score
import helper_proc as hp
import config_analysis
# =============================================================================
# Load data 
# =============================================================================
# =============================================================================
# Load data 
# =============================================================================
repository = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
data_path = os.path.join(repository, 'derivatives', 'ratings')
save_directory = os.path.join(repository, 'derivatives', 'LR')

for dummy in ['', 'dummy/']: #
    print(dummy)
    for setting in [ 'theory-based', 'full', 'data-driven']: #
   
        print(setting)
        df = pd.read_csv(os.path.join(data_path, 'preprocessed_features.csv'), decimal=',', sep=';')
        save_path = os.path.join(save_directory, setting, dummy)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        condition_list = df.loc[:, 'condition']
        audio_list = df.loc[:, 'audio']
        
        if setting == 'theory-based':
            drop = [col for col in df.columns if col not in ['valence_mean', 'arousal_mean', 'dominance']]
        elif (setting == 'full') or (setting == 'data-driven'):    
            drop = ['condition','audio', 'subj']
        
        df = df.drop(drop, axis = 1)
        df = df.apply(pd.to_numeric)
            
        rkf = RepeatedKFold(n_splits=5, n_repeats=100, random_state=42)
        for repeat, (train_index, test_index) in enumerate(rkf.split(np.array(df))):
            if repeat%100==0:
                print('{:.2f}%'.format((repeat*100)/(rkf.n_repeats * rkf.cvargs['n_splits'])),end=', ')
                     
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = np.array(df)[train_index], np.array(df)[test_index]
            y_train, y_test = np.array(condition_list)[train_index], np.array(condition_list)[test_index]

            if setting == 'data-driven':
                sfs = SFS(LogisticRegression(random_state = 42), k_features='best', forward=True, floating=True, verbose=1, scoring='accuracy', cv=5)
                sfs = sfs.fit(X_train, y_train)
                feature_idx = sfs.k_feature_idx_
                # Feature selection on data set
                X_train = X_train[:, list(feature_idx)]
                X_test = X_test[:, list(feature_idx)] 
                features = df.iloc[:, list(feature_idx)].columns
                SFFS_score = sfs.k_score_
            else: 
                features = df.columns
                SFFS_score = None
            if 'dummy' in dummy: 
                np.random.shuffle(y_train)
                np.random.shuffle(y_test)
            clf = LogisticRegression(random_state = 42).fit(X_train, y_train)
            df_train_prob = pd.DataFrame(clf.predict_proba(X_train))
            df_train_prob.columns = clf.classes_
            df_train_prob['audio'] = np.array(audio_list)[train_index]
            df_train_prob['condition'] = np.array(condition_list)[train_index]
            df_test_prob = pd.DataFrame(clf.predict_proba(X_test))
            df_test_prob.columns = clf.classes_
            df_test_prob['audio'] = np.array(audio_list)[test_index]
            df_test_prob['condition'] = np.array(condition_list)[test_index]
            df_full = df_train_prob.append(df_test_prob)
            df_full['fold'] = repeat
            
            accuracy = cross_val_score(LogisticRegression(random_state = 42), X_test, y_test, cv=5,scoring='accuracy')
            f1 = cross_val_score(LogisticRegression(random_state = 42), X_test, y_test, cv=5,scoring='f1_weighted')
            
            for a,f in zip(accuracy,f1):
                ml_score = pd.DataFrame({'Features' : [list(features)],
                                     'Score_SFFS' : [SFFS_score],
                                     'Test_accuracy' : [np.round(a,3)],
                                     'Test_f1' : [np.round(f,3)],
                                     'Repeat': [repeat]})        
                ml_score.to_csv(os.path.join(save_path, 'LR_performance.csv'), index = False, decimal = ',', sep = ';', mode = 'a', header = (not os.path.exists(os.path.join(save_path, 'LR_performance.csv'))))            
            df_full.to_csv(os.path.join(save_path, 'LR_probability.csv'), index = False, decimal = ',', sep = ';', mode = 'a', header = (not os.path.exists(os.path.join(save_path, 'LR_probability.csv'))))     

        print(setting)
        df = pd.read_csv(os.path.join('results','LR',setting , dummy + 'LR_probability.csv'), decimal=',', sep=';', header = 0)
        if 'audio_files' in df.columns: 
            df = df.rename(columns={'audio_files': 'audio'})
        conditions = sorted(list(df.condition.unique()))
        df['labels'] = [conditions[np.where(row == np.max(row))[0][0]] for row in np.array(df.loc[:,conditions])]
        df['critical'] = 0   
        #Get wrong classified samples
        for cond in conditions:
            temp = df.loc[df['condition'] == cond]
            df_temp = df.loc[(df['condition'] == cond) & (df['labels'] != temp['labels'].value_counts().index[0])]
            try:    
                df.loc[list(df_temp.index), 'critical'] = 1
            except:         
                print('df_temp empty')
        del temp

        hp.bootstrap_prob_LR(df, save_path)


