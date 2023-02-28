
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
import helper_proc as hp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
from fcmeans import FCM
from mpl_toolkits.mplot3d import Axes3D
import helper_plotting as pb
import config_analysis
import matplotlib as mpl
mpl.rc('font',family='Times New Roman', weight = 'bold')
# =============================================================================
# Load data 
# =============================================================================
repository = os.path.join(config_analysis.project_root, "gaudie_audio_validation")
data_path = os.path.join(repository, 'derivatives', 'ratings')
clustering = False
interactive_3D_plot = True
lst_color = ['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet']
dict_color = {'positive': lst_color[2],
              'neutral' : lst_color[0],
              'negative' : lst_color[1]}
# =============================================================================
# Clustering
# =============================================================================
if clustering:
    for setting in ['theory-based',  'data-driven']: #'full',
        print(setting)
        df = pd.read_csv(os.path.join(data_path,'preprocessed_features.csv'), decimal=',', sep=';')
        save_path = os.path.join(repository, 'derivatives', 'clustering', setting)
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

        if (setting == 'theory-based') or (setting == 'full'):
            columns = df.columns
            hp.elbow_plot(df, 10, save_path)
            hp.silhouette_plot(df, 5, save_path)

            # =============================================================================
            # Fitting the Real Model
            # =============================================================================

            fcm = FCM(n_clusters=3, random_state=42)
            fcm.fit(np.array(df))

            fcm_centrioids = fcm.centers
            with open(os.path.join(save_path, 'fcm_centrioids.pickle'), 'wb') as file:
                pickle.dump(fcm_centrioids, file, protocol=pickle.HIGHEST_PROTOCOL)
            fcm_labels  = fcm.predict(np.array(df))
            df['fcm_labels'] = fcm_labels
            df['condition'] = condition_list
            df['audio'] = audio_list
            df, clu_con_dict = hp.cluster_condition_dict(df, fcm)
            with open(os.path.join(save_path, 'clu_con_dict.pickle'), 'wb') as file:
                pickle.dump(clu_con_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
            if (setting == 'theory-based'):
                hp.scatterplot_2D(df, 'valence_mean', 'arousal_mean', fcm_labels, fcm_centrioids, clu_con_dict, save_path)
                hp.scatterplot_3D(df, columns[0], columns[1], columns[2], fcm_labels, fcm_centrioids, columns, clu_con_dict, save_path)
                #hp.scatterplot_3D_interactive(df, columns[0], columns[1], columns[2], fcm_labels, fcm_centrioids, columns, clu_con_dict, save_path)
                hp.PCA_features(df, columns, 3, save_path, plot_2_comp = True, plot_3_comp = True, plot_fcm_pca = True)

                # =============================================================================
                # Interactive Plot
                # =============================================================================
                titles = ['Dominance', 'Arousal', 'Valence']
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                scatter = ax.scatter(xs = df[columns[0]] , ys = df[columns[1]] , zs = df[columns[2]] , c=fcm_labels, alpha=.7,cmap = 'jet_r')
                ax.scatter(xs = fcm_centrioids[:,list(df.columns).index(columns[0])], ys = fcm_centrioids[:,list(df.columns).index(columns[1])], zs = fcm_centrioids[:,list(df.columns).index(columns[2])],
                           marker="+", s=100, c='black', label='Centroids')
                ax.set_xlabel(titles[0], fontsize = 12, fontweight = 'bold')
                ax.set_xticklabels(np.round(ax.get_xticks(),2), fontsize = 10, c = 'black')
                ax.set_ylabel(titles[1], fontsize = 12, fontweight = 'bold')
                ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 10, c = 'black')
                ax.set_zlabel(titles[2], fontsize = 12, fontweight = 'bold')
                ax.set_zticklabels(np.round(ax.get_zticks(),2), fontsize = 10, c = 'black')
                fig.suptitle('Clustered Audio Segments', fontsize = 14, fontweight = 'bold')
                fig.legend(handles=scatter.legend_elements()[0], labels=list(clu_con_dict.keys()), loc='upper right')
                plt.tight_layout()

            # =============================================================================
            # Bootstrapping Probability for Critical, Pos, Neg, Neutral
            # =============================================================================
            hp.bootstrap_prob(df, save_path, columns)
            # =============================================================================
            # Heatmap
            # =============================================================================
            df, df_heat_sorted = hp.distance_heatmap(df, fcm, columns, 'euclidean', save_path)
            # =============================================================================
            # Save Results
            # =============================================================================
            df.to_csv(os.path.join(save_path, 'df_clustering.csv'), index = False, header = True, decimal = ',', sep = ';')
            df_heat_sorted.to_csv(os.path.join(save_path, 'df_distance_euclidean.csv'), index = False, header = True, decimal = ',', sep = ';')
            # =============================================================================
            # Accuracy
            # =============================================================================
            accuracy = 100 - (len(df) / df['critical'].value_counts()[1])
            accuracy_file = open(os.path.join(save_path,"accuracy_clustering.txt"),'w')
            accuracy_file.write(accuracy)
            accuracy_file.close()
        if setting == 'data-driven':
            best_of, result_dict = hp.permutation_cluster_feature(df, list(range(0, len(df.columns))), 10, condition_list)
            best_of.to_csv(os.path.join(save_path, 'best_models.csv'), index = False, header = True, decimal = ',', sep = ';')
            with open(os.path.join(save_path, 'result_dict.pickle'), 'wb') as file:
                pickle.dump(result_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
            for metric_score, cols in [('best_model_strength','best_set_model_strength'),
                           ('best_accuracy_score','best_set_accuracy_score'),
                           ('best_f1_score','best_set_f1_score')]:
                print(best_of[metric_score])
                save_metric = os.path.join(save_path, metric_score[5:])
                if not os.path.exists(save_metric):
                    os.makedirs(save_metric)
                columns = best_of[cols].values[0]
                df_clustering = df.loc[:,columns]

                hp.elbow_plot(df_clustering, 10, save_metric)
                hp.silhouette_plot(df_clustering, 5, save_metric)

                # =============================================================================
                # Fitting the Real Model
                # =============================================================================

                fcm = FCM(n_clusters=3, random_state=42)
                fcm.fit(np.array(df_clustering))
                fcm_centrioids = fcm.centers
                with open(os.path.join(save_path, 'fcm_centrioids.pickle'), 'wb') as file:
                    pickle.dump(fcm_centrioids, file, protocol=pickle.HIGHEST_PROTOCOL)
                fcm_labels  = fcm.predict(np.array(df_clustering))
                df_clustering['fcm_labels'] = fcm_labels
                df_clustering['condition'] = condition_list
                df_clustering['audio'] = audio_list
                df_clustering, clu_con_dict = hp.cluster_condition_dict(df_clustering, fcm)

                with open(os.path.join(save_path, 'clu_con_dict.pickle'), 'wb') as file:
                    pickle.dump(clu_con_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
                # =============================================================================
                # Bootstrapping Probability for Critical, Pos, Neg, Neutral
                # =============================================================================
                hp.bootstrap_prob(df_clustering, save_metric, columns)
                # =============================================================================
                # Heatmap
                # =============================================================================
                df_clustering, df_heat_sorted = hp.distance_heatmap(df_clustering, fcm, columns, 'euclidean', save_metric)
                # =============================================================================
                # Save Results
                # =============================================================================
                df_clustering.to_csv(os.path.join(save_metric, 'df_clustering.csv'), index = False, header = True, decimal = ',', sep = ';')
                df_heat_sorted.to_csv(os.path.join(save_metric, 'df_distance_euclidean.csv'), index = False, header = True, decimal = ',', sep = ';')

                # =============================================================================
                # Accuracy
                # =============================================================================
                accuracy = 100 - (len(df_clustering) / df_clustering['critical'].value_counts()[1])
                accuracy_file = open(os.path.join(save_metric,"accuracy_clustering.txt"),'w')
                accuracy_file.write(accuracy)
                accuracy_file.close()

# =============================================================================
# Interactive Plotting  for V A D
# =============================================================================
only_legend = False
if interactive_3D_plot:
    if os.path.exists(os.path.join(repository, 'derivatives', 'clustering', 'theory-based', 'df_clustering.csv')):
        df = pd.read_csv(os.path.join(repository, 'derivatives', 'clustering', 'theory-based', 'df_clustering.csv'), decimal=',', sep=';')
        fcm_labels = df['fcm_labels']
        fcm_centrioids = pd.read_pickle(os.path.join(repository, 'derivatives', 'clustering', 'theory-based', 'fcm_centrioids.pickle'))
        clu_con_dict = pd.read_pickle(os.path.join(repository, 'derivatives', 'clustering', 'theory-based', 'clu_con_dict.pickle'))

        titles = ['Dominance', 'Arousal', 'Valence']
        columns = df.columns[:len(titles)]
        fig = plt.figure()
        if only_legend:
            ax = fig.add_subplot(projection='3d')
            color_list = fcm_labels.replace({clu_con_dict['negative']:dict_color['negative'],
                                             clu_con_dict['neutral']:dict_color['neutral'],
                                             clu_con_dict['positive']:dict_color['positive']})
            scatter = ax.scatter(xs = df[columns[0]] , ys = df[columns[1]] , zs = df[columns[2]] , c=color_list, alpha=.7)
            ax.scatter(xs = fcm_centrioids[:,list(df.columns).index(columns[0])], ys = fcm_centrioids[:,list(df.columns).index(columns[1])], zs = fcm_centrioids[:,list(df.columns).index(columns[2])],
                       marker="+", s=100, c='black', label='Centroids')
            ax.set_xlabel(titles[0], fontsize = 12, fontweight = 'bold')
            ax.set_xticklabels(np.round(ax.get_xticks(),2), fontsize = 10, c = 'black')
            ax.set_ylabel(titles[1], fontsize = 12, fontweight = 'bold')
            ax.set_yticklabels(np.round(ax.get_yticks(),2), fontsize = 10, c = 'black')
            ax.set_zlabel(titles[2], fontsize = 12, fontweight = 'bold')
            ax.set_zticklabels(np.round(ax.get_zticks(),2), fontsize = 10, c = 'black')
            #fig.suptitle('Clustered Audio Segments', fontsize = 14, fontweight = 'bold')
        fig.legend([Line2D([0], [0], color=dict_color['negative'], marker='o', linestyle="None"),
          Line2D([0], [0], color=dict_color['neutral'], marker='o', linestyle="None"),
          Line2D([0], [0], color=dict_color['positive'], marker='o', linestyle="None")],
          ['negative', 'neutral','positive'], loc='upper right')
        plt.tight_layout()
        plt.show()
