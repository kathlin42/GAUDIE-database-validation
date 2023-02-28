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
import helper_plotting as pb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import config_analysis
# =============================================================================
# Load data 
# =============================================================================
bids_root = os.path.join(config_analysis.project_root, "gaudie_audio_validation")

data_path = os.path.join(bids_root, 'derivatives', 'LR')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for dummy in ['', 'dummy/']:
    for setting in ['theory-based', 'full', 'data-driven']:
        print(setting)
        df = pd.read_csv(os.path.join(data_path, setting, dummy + 'LR_performance.csv'), decimal=',', sep=';', header = 0)
        df_group = df.groupby('Features').count().iloc[:,0].sort_values(ascending = False)
        max_len = 0
        for idx, feat_list in enumerate(df['Features']):
            
            feat_list = feat_list.rsplit(',')
            feat_list = [feature.replace('[', '') for feature in feat_list]
            feat_list = [feature.replace(' ', '') for feature in feat_list]
            feat_list = [feature.replace(']', '') for feature in feat_list]
            if max_len < len(feat_list):
                max_len = len(feat_list)
            df.loc[idx, 'Features'] = str(sorted(feat_list))
        df_group_check = df.groupby('Features').count().iloc[:,0].sort_values(ascending = False)
        df_group_check = df_group_check.reset_index()
        #df_group_check.to_csv(os.path.join(data, setting, 'Feature_combinations.csv'), decimal=',', sep=';', index = True, header = True)
        df_LR = pd.DataFrame()
        for metric in ['Test_accuracy','Test_f1']:
            dict_boot = pb.bootstrapping(np.array(df[metric]),
                             sample_size = len(np.array(df['Test_accuracy'])),
                             numb_iterations = 5000,
                             alpha =0.95,
                             plot_hist =True,
                             as_dict = True,
                             func = 'mean')
            df_LR = df_LR.append(pd.DataFrame(dict_boot, index = [metric]))
            
        plt.suptitle('Distribution of Performance over 500 Folds - ' + setting, fontsize = 13, fontweight = 'bold')
        plt.legend([Line2D([0], [0], color=colors[0], lw=3), 
                    Line2D([0], [0], color=colors[1], lw=3)], 
                    [['Test_accuracy','Test_f1'][0], 
                     ['Test_accuracy','Test_f1'][1]], 
                    loc ='upper right', 
                    facecolor='white',
                    fontsize='medium')
        plt.xlabel('Performance Metric', fontsize = 12, fontweight = 'bold')
        plt.ylabel('Frequency', fontsize = 12, fontweight = 'bold')
        plt.tight_layout()
        plt.savefig(os.path.join(data_path, setting, dummy + 'Distribution_performance.svg'))
        plt.close()
        df_LR.to_csv(os.path.join(data_path, setting, dummy + 'LR_evaluation.csv'), index = True, header = True, decimal=',', sep=';' )
        
            
