# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 19:13:17 2022

@author: hirning
"""

import numpy as np
import pandas as pd
import os
import pickle
from math import sqrt
from matplotlib.lines import Line2D
import helper_plotting as pb
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import pingouin as pg
import config_analysis
# =============================================================================
# Load data 
# =============================================================================
repository = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
data_path = os.path.join(repository, 'derivatives', 'clustering')

settings = [os.path.join('data-driven', 'accuracy_score', 'Evaluation'), os.path.join('theory-based','Evaluation')]
columns = ['ID', "* 2.5th CI","* M","* 97.5th CI", "^^ 2.5th CI","^^ M","^^ 97.5th CI", "$ 2.5th CI","$ M","$ 97.5th CI"]
for i, setting in enumerate(settings):
    bootstrapped_files = [file for file in os.listdir(os.path.join(data_path,setting)) if 'csv' in file]
    df_full = pd.DataFrame(columns =columns)

    for ii, file in enumerate(bootstrapped_files):
        df = pd.read_csv(os.path.join(data_path, setting, file), decimal=',', sep=';', header = 0)
        listed_values = [[val] for val in df.iloc[:,1:].values.flatten()]
        bootstrapping = pd.DataFrame(dict(zip(columns, [[df.columns[0]]] + listed_values)))
        df_full = df_full.append(bootstrapping)
    df_full.columns = ['ID'] + ["2.5th CI","M","97.5th CI"] * 3
    df_full.to_csv(os.path.join(data_path,os.path.split(setting)[0], 'critical_bootstrapped_values_full.csv'), index = False, header = True, decimal=',', sep=';' )
