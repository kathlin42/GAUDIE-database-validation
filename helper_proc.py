# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 19:47:15 2021

@author: hirning
"""
import os
import numpy as np
import itertools
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from fcmeans import FCM
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import helper_plotting as pb
import seaborn as sns
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import paired_distances

# =============================================================================
# Permutation Feature in Cluster Analysis
# =============================================================================
def permutation_cluster_feature(df, idx_cols, num_top, condition_list):
    result_dict = {}
    ID = 0
    best_set_ID_model_strength = None
    best_set_model_strength = None
    best_model_strength = 0
    Top_10_model_strength = []
    
    best_set_ID_f1_score = None
    best_set_f1_score = None
    best_f1_score = 0
    Top_10_f1_score = []
    
    best_set_ID_accuracy_score = None
    best_set_accuracy_score = None
    best_accuracy_score = 0
    Top_10_accuracy_score = []
    
    iterations = 0
    for r in range(3,len(idx_cols)+1):
        permutations = list(itertools.combinations(idx_cols, r))
        iterations = iterations + len(permutations)
         
    for r in range(3,len(idx_cols)+1):
        permutations = list(itertools.combinations(idx_cols, r))
        for element in permutations:
            ID = ID + 1
            if ID%100==0:
                print('{:.2f}%'.format(ID*100/iterations),end=', ')
                
            col_names = [col for idx, col in enumerate(df.columns) if idx in element]
            #print('List_combi', col_names)
            df_clu = df.loc[:, col_names]
            
            
            fcm = FCM(n_clusters=3, random_state=42)
            fcm.fit(np.array(df_clu))
            model_strength = sum([np.max(i) for i in fcm.u])
            df_clu['fcm_labels']  = fcm.predict(np.array(df_clu))
            
            # =============================================================================
            # Get relationship Cluster - Condition
            # =============================================================================
            dict_con_clu = {}
            df_clu['condition'] = condition_list
            df_clu['critical'] = 0
            
            for cond in df_clu['condition'].unique():
                temp = df_clu.loc[df_clu['condition'] == cond]
                dict_con_clu[cond] = temp['fcm_labels'].value_counts().index[0]
                df_temp = df_clu.loc[(df_clu['condition'] == cond) & (df_clu['fcm_labels'] != temp['fcm_labels'].value_counts().index[0])]
                try:    
                    df_clu.loc[list(df_temp.index), 'critical'] = 1
                except:         
                    print('df_temp empty')
            
            unsorted_dict = sorted(dict_con_clu, key=dict_con_clu.get)
            sorted_dict = {}
            for key in unsorted_dict:
                sorted_dict[key] = dict_con_clu[key]
            df_clu['condition'] = df_clu['condition'].replace(sorted_dict)
            f1 = f1_score(df_clu['condition'], df_clu['fcm_labels'], average = 'weighted')
            accuracy = accuracy_score(df_clu['condition'], df_clu['fcm_labels'])
            result_dict[ID] = {'r':r,
                              'col_names' : col_names,
                              'model_strength' : np.round(model_strength,3),
                              'f1' : np.round(f1, 3),
                              'accuracy':np.round(accuracy, 3)} 
            
            if model_strength >= best_model_strength:
                best_model_strength = model_strength
                best_set_model_strength = col_names
                best_set_ID_model_strength = ID
            if len(Top_10_model_strength) < num_top:
                Top_10_model_strength.insert(0,ID)
            elif any(model < model_strength for model in Top_10_model_strength): 
                Top_10_model_strength.sort()
                Top_10_model_strength.pop(0)
                Top_10_model_strength.insert(0,ID)
                    
            if f1 >= best_f1_score:
                best_f1_score  = f1
                best_set_f1_score = col_names
                best_set_ID_f1_score = ID
            if len(Top_10_f1_score) < num_top:
                Top_10_f1_score.insert(0,ID)
            elif any(model < f1 for model in Top_10_f1_score): 
                Top_10_f1_score.sort()
                Top_10_f1_score.pop(0)
                Top_10_f1_score.insert(0,ID)
                
            if accuracy >= best_accuracy_score:
                best_accuracy_score = accuracy
                best_set_accuracy_score = col_names
                best_set_ID_accuracy_score = ID
            if len(Top_10_accuracy_score) < num_top:
                Top_10_accuracy_score.insert(0,ID)
            elif any(model < accuracy for model in Top_10_accuracy_score): 
                Top_10_accuracy_score.sort()
                Top_10_accuracy_score.pop(0)
                Top_10_accuracy_score.insert(0,ID)
    
    best_of = pd.DataFrame({'best_model_strength':[best_model_strength],
                  'best_set_model_strength':[best_set_model_strength],
                  'best_set_ID_model_strength':[best_set_ID_model_strength],
                  'Top_10_model_strength':[Top_10_model_strength],
                  'best_accuracy_score':[best_accuracy_score],
                  'best_set_accuracy_score':[best_set_accuracy_score],
                  'best_set_ID_accuracy_score':[best_set_ID_accuracy_score],
                  'Top_10_accuracy_score':[Top_10_accuracy_score],
                  'best_f1_score':[best_f1_score],
                  'best_set_f1_score':[best_set_f1_score],
                  'best_set_ID_f1_score':[best_set_ID_f1_score],
                  'Top_10_f1_score':[Top_10_f1_score]})
    
    return best_of, result_dict

# =============================================================================
# Elbow Method
# =============================================================================

def elbow_plot(df, max_k, save_path):
    scores = []
    for cluster in range(1, max_k):
        fitted_model = FCM(n_clusters=cluster, random_state=42)
        fitted_model.fit(np.array(df))
        scores.append(sum([np.max(i) for i in fitted_model.u]))
    plt.figure(figsize=(10,5))
    plt.plot(range(1, max_k), scores, "bx-")
    plt.title('Elbow Method using Aggregated Fuzzy Partition Coefficient (FPC)', fontsize=14, fontweight='bold')
    plt.xlabel('k')
    plt.ylabel('Aggregated Fuzzy Partition Coefficient (FPC)')
    plt.xticks(range(1, max_k))
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Elbow_Method_FCM.svg'))
    plt.close()

# =============================================================================
# Silhouette Score
# =============================================================================
def silhouette_plot(df, max_k, save_path):
    for clusters in range(2, max_k):
        fitted_model = FCM(n_clusters=clusters, random_state=42)
        fitted_model.fit(np.array(df))
        model_labels = fitted_model.u.argmax(axis=1)
        silhouette_avg = silhouette_score(df, model_labels)
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(df, model_labels)
        plt.figure(figsize=(10, 5))
        y_lower = 10
        for cluster in range(clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[model_labels == cluster]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(cluster) / clusters)
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            plt.text(0.05, y_lower + 0.5 * size_cluster_i, 'Cluster ' + str(cluster))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10

        plt.title('K = ' + str(clusters) + ', Average Silhouette Score = ' + str(round(silhouette_avg, 2)))
        plt.xlabel('Silhouette Coefficient Values')
        plt.ylabel('Cluster Labels')
    
        # The vertical line for average silhouette score of all the values
        plt.axvline(x=silhouette_avg, color='red', linestyle='--')
    
        plt.suptitle('Silhouette Score for Amount of FCM-Cluster',
                     fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(save_path, 'Silhouette_Analysis_FCM_K=' + str(clusters) + '.svg'))
        plt.show()
        plt.close()

# =============================================================================
# Getting Cluster-Condition Dict
# =============================================================================

def cluster_condition_dict(df, fcm):
    dict_con_clu = {}
    sorted_dict = {}
    df['critical'] = 0      
    for cond in sorted(list(df.condition.unique())):
        temp = df.loc[df['condition'] == cond]
        df['prob_' +  cond]  = fcm.u[:,temp['fcm_labels'].value_counts().index[0]]
        dict_con_clu[cond] = temp['fcm_labels'].value_counts().index[0]
        df_temp = df.loc[(df['condition'] == cond) & (df['fcm_labels'] != temp['fcm_labels'].value_counts().index[0])]
        try:    
            df.loc[list(df_temp.index), 'critical'] = 1
        except:         
            print('df_temp empty')
    
    unsorted_dict = sorted(dict_con_clu, key=dict_con_clu.get)
    for key in unsorted_dict:
        sorted_dict[key] = dict_con_clu[key]
    return df, sorted_dict

# =============================================================================
# 2D Scatterplot  
# =============================================================================

def scatterplot_2D(df, var1, var2, fcm_labels, centroids, cluster_condition_dict, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    scatter = ax.scatter(df[var1] , df[var2] , c=fcm_labels, alpha=.5)
    ax.scatter(centroids[:,list(df.columns).index(var1)], centroids[:,list(df.columns).index(var2)], marker="+", s=100, c='black', label='Centroids')
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    plt.title('Scatterplot of the FCM Clusters', fontsize = 14)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(cluster_condition_dict.keys()), loc='upper right')
    plt.savefig(os.path.join(save_path,'scatter_'+ var1 + '-'+ var2 + '.svg'))
    plt.show()
    plt.close()

# =============================================================================
# 3D Scatterplot  
# =============================================================================

def scatterplot_3D(df, var1, var2, var3, fcm_labels, centroids, axes_labels, cluster_condition_dict, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    scatter = ax.scatter(xs = df[var1] , ys = df[var2] , zs = df[var3] , c=fcm_labels, alpha=.5)
    ax.scatter(xs = centroids[:,list(df.columns).index(var1)], ys = centroids[:,list(df.columns).index(var2)], zs = centroids[:,list(df.columns).index(var3)], marker="+", s=100, c='black', label='Centroids')
    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    ax.set_zlabel(axes_labels[2])
    fig.suptitle('Scatterplot of the FCM Clusters', fontsize = 14)
    fig.legend(handles=scatter.legend_elements()[0], labels=list(cluster_condition_dict.keys()), loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(save_path,'scatter_'+ var1 + '-'+ var2 + '-' + var3 + '.svg'))
    plt.close(fig)
    

    
    
# =============================================================================
# PCA to 17 dimensions
# =============================================================================
def PCA_features(df, features, n_components, save_path, plot_2_comp = False, plot_3_comp = False, plot_fcm_pca = False):
    pca = PCA(n_components=n_components, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=42)
    feat_pca = pca.fit_transform(df.loc[:, features])
    n_components = feat_pca.shape[1]
    columns = ['PCA_'+str(i) for i in list(range(1,n_components+1))]
    df_feat_pca = pd.DataFrame(feat_pca, columns=columns)
  
    for idx, col in enumerate(columns): 
        df[col] = feat_pca[:,idx]
    
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))   
    
    if plot_2_comp:
        labels = np.array(df.condition.replace({'negative': 0, 'neutral': 1, 'positive': 2 }))
        fig = plt.figure(figsize=(12, 10), facecolor='white')   
        fig.suptitle('Visualizing Projections of Audiosegments', fontsize = 20, fontweight='bold')
        
        for n_comp, (comp_1, comp_2) in enumerate(zip([0,1,2],[1,2,0])):
            ax = fig.add_subplot(2,2, n_comp + 1)
            scatter = ax.scatter(feat_pca[:, comp_1], feat_pca[:, comp_2],
                        c=labels, edgecolor='none', alpha=0.5,
                        cmap=plt.cm.get_cmap('Accent', 3))
            ax.set_xlabel('PCA Component ' + str(comp_1 +1))
            ax.set_ylabel('PCA Component ' + str(comp_2 +1))
            cbar = fig.colorbar(scatter, ticks = [0.25,1,1.75])
            cbar.ax.set_yticklabels(['negative', 'neutral', 'positive'], rotation = 90, va='center')
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, 'PCA_projections_seperate_vis.svg'))   
        plt.close(fig)
        
    if plot_3_comp:
        labels = np.array(df.condition.replace({'negative': 0, 'neutral': 1, 'positive': 2 }))
        fig = plt.figure(figsize=(12, 10), facecolor='white')   
        fig.suptitle('Visualizing Projections of Audiosegments', fontsize = 20, fontweight='bold')
        ax = fig.add_subplot(projection='3d')
        scatter = ax.scatter(xs = feat_pca[:, 0], ys = feat_pca[:, 1], zs = feat_pca[:, 2],
                    c=labels, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('Accent', 3))
        ax.set_xlabel('PCA Component ' + str(0 + 1))
        ax.set_ylabel('PCA Component ' + str(1 + 1))
        ax.set_zlabel('PCA Component ' + str(2 + 1))
        cbar = fig.colorbar(scatter, ticks = [0.25,1,1.75])
        cbar.ax.set_yticklabels(['negative', 'neutral', 'positive'], rotation = 90, va='center')
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, 'PCA_projections_3d_vis.svg'))   
        plt.close(fig)
    
    if plot_fcm_pca:
        pca = PCA(n_components=2, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=42)
        feat_pca = pca.fit_transform(df.loc[:, features])
        n_components = feat_pca.shape[1]
        columns = ['PCA_'+str(i) for i in list(range(1,n_components+1))]
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))   
        pca_fcm = FCM(n_clusters=3, random_state=42)
        pca_fcm.fit(feat_pca)
        pca_fcm_centrioids = pca_fcm.centers
        pca_fcm_labels = pca_fcm.predict(feat_pca)
        df_pca = df.copy()
        df_pca['fcm_labels'] = pca_fcm_labels
        # =============================================================================
        # Scaling the boundaries
        # =============================================================================
        xs = feat_pca[:,0]
        ys = feat_pca[:,1]
        n = pca.components_.shape[0]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())
        
        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
        
        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = (xs * scalex).min() - 1, (xs * scalex).max() + 1
        y_min, y_max = (ys * scaley).min() - 1, (ys * scaley).max() + 1
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        pca_fcm_centers_x, pca_fcm_centers_y = (pca_fcm_centrioids[:,0] * scalex), (pca_fcm_centrioids[:,1] * scaley)
        
        # Obtain labels for each point in mesh. Use last trained model.
        model_boundaries = FCM(n_clusters=3, random_state=42)
        model_boundaries.fit(np.c_[xx.ravel(), yy.ravel()])
        Z = model_boundaries.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap='plasma_r', alpha = 0.3,
               aspect='auto', origin='lower')
        plt.show()
        scatter = plt.scatter(xs * scalex,ys * scaley, c=pca_fcm_labels)
        plt.scatter(pca_fcm_centers_x, pca_fcm_centers_y, marker="+", s=100, c='black', label='centroids')
        labels = None
        for i in range(n):
            plt.arrow(0, 0, pca.components_[i,0], pca.components_[i,1],color = 'r',alpha = 0.5, shape = 'full', width = 0.001)
            if labels is None:
                plt.text(pca.components_[i,0] * 1.15, pca.components_[i,1] * 1.15 +0.08, "Var "+ str(i+1), color = 'r', ha = 'center', va = 'center')
            else:
                plt.text(pca.components_[i,0]* 1.15, pca.components_[i,1] * 1.15, labels[i], color = 'r', ha = 'center', va = 'center')
        df_pca, sorted_dict = cluster_condition_dict(df_pca, pca_fcm)
        plt.legend(handles=scatter.legend_elements()[0], labels=list(sorted_dict.keys()), loc='upper right')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xlabel("PC{}".format(1))
        plt.ylabel("PC{}".format(2))
        plt.title('Scatterplot of the PCA-based FCM Clusters', fontsize = 14)
        plt.savefig(os.path.join(save_path, 'PCA-cluster-based_scatterplot.svg'))
        plt.close() 
    return df_feat_pca

def bootstrap_prob(df, save_path, cols):
    # =============================================================================
    # Bootstrapping Probability for Critical, Pos, Neg, Neutral
    # =============================================================================
    # Plotting the components
    df_critical = df.loc[df['critical'] == 1]
    df_pos = df.loc[(df['critical'] == 0) & (df['condition'] == 'positive')]
    df_neg = df.loc[(df['critical'] == 0) & (df['condition'] == 'negative')]
    df_neu = df.loc[(df['critical'] == 0) & (df['condition'] == 'neutral')]
    if not os.path.exists(os.path.join(save_path, 'Bootstrapping_Prob')):
        os.makedirs(os.path.join(save_path, 'Bootstrapping_Prob'))
    if not os.path.exists(os.path.join(save_path, 'Evaluation')):
        os.makedirs(os.path.join(save_path, 'Evaluation'))

    for title, df_plot in zip(['Critical Audiofiles', 'Positive Audiofiles', 'Negative Audiofiles', 'Neutral Audiofiles'], [df_critical, df_pos, df_neg, df_neu]):
        pb.plot_boxplots(sorted(list(df_plot['audio'].unique())),
                         df_plot,
                         'audio',
                         ['prob_negative', 'prob_neutral','prob_positive'] + list(cols), 
                         boot = 'mean', 
                         boot_size = 5000,
                         title=title, 
                         lst_color=None, 
                         save_path = os.path.join(save_path, 'Bootstrapping_Prob', title[:-11]), 
                         fwr_correction = False)
    
        plt.savefig(os.path.join(save_path, 'Bootstrapping_Prob' , 'Bootstrapping_' + title[:-11] + '.svg'))
        plt.close() 
       
    # =============================================================================
    # Critical Files Evaluation - Probability Score
    # =============================================================================
    df_critical_files = df.loc[df.audio.isin(df_critical.audio.unique()),:]
    # =============================================================================
    # Bootstrapping all Critical Files
    # =============================================================================
    pb.plot_boxplots(sorted(list(df_critical.audio.unique())),
                      df_critical_files,
                     'audio',
                     ['prob_negative', 'prob_neutral','prob_positive'] + list(cols), 
                     boot = 'mean', 
                     boot_size = 5000,
                     title='All Responses for Critical Audiofiles', 
                     lst_color=None, 
                     save_path = os.path.join(save_path, 'Bootstrapping_Prob', 'All_Critical'), 
                     fwr_correction = False)

    plt.savefig(os.path.join(save_path, 'Bootstrapping_Prob' , 'Bootstrapping_all_critical.svg'))
    plt.close()
    
    # =============================================================================
    # Evaluation
    # =============================================================================        

    for cond in list(df_critical_files.condition.unique()):
        df_plot = df_critical_files.loc[df_critical_files.condition == cond,['audio','prob_negative', 'prob_neutral','prob_positive']].melt(id_vars =['audio'])
        df_plot['audio'] = [row.split('_')[0] + ' ' + str(int(row.split('_')[1])+1) for row in df_plot['audio']]
        df_plot['variable'] = [row.replace('_', ' ') for row in df_plot['variable']]
        df_plot['variable'] = [row.replace('p', 'P') for row in df_plot['variable']]
        df_plot['variable'] = [row.replace('n', 'N') for row in df_plot['variable']]
        df_plot['variable'] = [row[:8] for row in df_plot['variable']]
        pb.plot_eva_audio(sorted(list(df_plot['variable'].unique())),
                         df_plot,
                         'variable',
                         'audio',
                         sorted(list(df_plot['audio'].unique())), 
                         boot = 'mean', 
                         boot_size = 5000,
                         title='Evaluation of Audiosegments', 
                         lst_color=None, 
                         save_path = os.path.join(save_path, 'Evaluation'), 
                         fwr_correction = False)
    
        plt.savefig(os.path.join(save_path, 'Evaluation' , 'Evaluation_' + cond + '.svg'))
        plt.close()
    
       
    # =============================================================================
    # Plotting Heatmap Distance
    # =============================================================================
def distance_heatmap(df, fcm, cols, metric = 'euclidean', save_path = ''):    
    if not os.path.exists(os.path.join(save_path, 'Distance')):
        os.makedirs(os.path.join(save_path, 'Distance'))
    dist = DistanceMetric.get_metric(metric)
    df['Dis_'+ metric + 'centroids'] = None
    
    
    for idx_sample, sample in enumerate(df['fcm_labels']):
        if idx_sample == 0:
            centroids = fcm.centers[sample,:]
        else: 
            centroids = np.vstack((centroids, fcm.centers[sample,:]))
    
    df['Dis_'+ metric + 'centroids'] = paired_distances(np.array(df.loc[:,cols]),centroids)
    # Plotting the components
    df_critical = df.loc[df['critical'] == 1]
    df_pos = df.loc[(df['critical'] == 0) & (df['condition'] == 'positive')]
    df_neg = df.loc[(df['critical'] == 0) & (df['condition'] == 'negative')]
    df_neu = df.loc[(df['critical'] == 0) & (df['condition'] == 'neutral')]
    
    for title, df_plot in zip(['Critical Audiofiles', 'Positive Audiofiles', 'Negative Audiofiles', 'Neutral Audiofiles'], [df_critical, df_pos, df_neg, df_neu]):
        pb.plot_boxplots(sorted(list(df_plot['audio'].unique())),
                         df_plot,
                         'audio',
                         ['Dis_'+ metric + 'centroids'], 
                         boot = 'mean', 
                         boot_size = 5000,
                         title=title, 
                         lst_color=None, 
                         save_path = os.path.join(save_path, 'Distance'), 
                         fwr_correction = False)
    
        plt.savefig(os.path.join(save_path, 'Distance', 'Dis_'+ metric + '_centroids_' + title[:-11] + '.svg'))
        plt.close()
        
    # =============================================================================
    # Plotting Heatmap Distance
    # =============================================================================
    for idx_sample, sample in enumerate(df['fcm_labels']):
        if idx_sample == 0:
            centroids = fcm.centers[sample,:]
        else: 
            centroids = np.vstack((centroids, fcm.centers[sample,:]))
    
    columns_heat = sorted(list(df.audio.unique()))
    df_heat_sorted =  pd.DataFrame(columns = columns_heat, index = columns_heat) 
    
    for start_audio in sorted(list(df.audio.unique())):
        for ref_audio in sorted(list(df.audio.unique())):
            #print(start_audio, ref_audio)
            distance = dist.pairwise(np.array(df.loc[:,cols])[list(np.where(df.audio == start_audio)[0])],np.array(df.loc[:,cols])[list(np.where(df.audio == ref_audio)[0])])
            #print(distance.mean())
            df_heat_sorted.loc[df_heat_sorted.index == ref_audio, start_audio] = distance.mean()
    df_heat_sorted = df_heat_sorted.apply(pd.to_numeric)
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(metric + ' Distance between Audio Segments', fontsize = 20)
    sns.heatmap(df_heat_sorted)
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, 'Distance', 'Dis_'+ metric + '_segments.svg'))
    plt.close() 
    
    positive_columns = [col for idx, col in enumerate(df_heat_sorted.columns) if 'positive' in col]
    negative_columns = [col for idx, col in enumerate(df_heat_sorted.columns) if 'negative' in col]
    neutral_columns = [col for idx, col in enumerate(df_heat_sorted.columns) if 'neutral' in col]
    positive_idx = [idx for idx, col in enumerate(df_heat_sorted.columns) if 'positive' in col]
    negative_idx = [idx for idx, col in enumerate(df_heat_sorted.columns) if 'negative' in col]
    neutral_idx = [idx for idx, col in enumerate(df_heat_sorted.columns) if 'neutral' in col]
    
    for title, tuple_cond in zip(['Positive Audiofiles', 'Negative Audiofiles', 'Neutral Audiofiles'], [(positive_columns,positive_idx), (negative_columns,negative_idx), (neutral_columns, neutral_idx)]):
        if not os.path.exists(os.path.join(save_path, 'Distance_per_Condition')):
            os.makedirs(os.path.join(save_path, 'Distance_per_Condition'))
        columns = tuple_cond[0]
        df_plot = pd.DataFrame()
        for col in columns:
            print(col)
            
            df_temp = df_heat_sorted.loc[col, columns]
            df_temp.index = [col] * len(df_temp.index)
            df_temp = df_temp.reset_index()
            df_temp = df_temp.rename(columns ={'index': 'Audio', col : 'Euclidean Distance'})
            df_plot = df_plot.append(df_temp, ignore_index=True)

        pb.plot_boxplots(sorted(list(df_plot['Audio'].unique())),
                         df_plot,
                         'Audio',
                         ['Euclidean Distance'], 
                         boot = 'mean', 
                         boot_size = 5000,
                         title=title, 
                         lst_color=None, 
                         save_path = os.path.join(save_path, 'Distance_per_Condition'), 
                         fwr_correction = False)
    
        plt.savefig(os.path.join(save_path, 'Distance_per_Condition', 'Euclidean_Distance_among_' + title[:-11] + '.svg'))
        plt.close()
    return df, df_heat_sorted

def bootstrap_prob_LR(df, save_path):
    # =============================================================================
    # Bootstrapping Probability for Critical, Pos, Neg, Neutral
    # =============================================================================
    # Plotting the components
    df_critical = df.loc[df['critical'] == 1]
    df_pos = df.loc[(df['critical'] == 0) & (df['condition'] == 'positive')]
    df_neg = df.loc[(df['critical'] == 0) & (df['condition'] == 'negative')]
    df_neu = df.loc[(df['critical'] == 0) & (df['condition'] == 'neutral')]
    
    if not os.path.exists(os.path.join(save_path, 'Bootstrapping_Prob')):
        os.makedirs(os.path.join(save_path, 'Bootstrapping_Prob'))
    if not os.path.exists(os.path.join(save_path, 'Evaluation')):
        os.makedirs(os.path.join(save_path, 'Evaluation'))

    for title, df_plot in zip(['Critical Audiofiles', 'Positive Audiofiles', 'Negative Audiofiles', 'Neutral Audiofiles'], [df_critical, df_pos, df_neg, df_neu]):
        pb.plot_boxplots(sorted(list(df_plot['audio'].unique())),
                         df_plot,
                         'audio',
                         ['negative', 'neutral', 'positive'], 
                         boot = 'mean', 
                         boot_size = 5000,
                         title=title, 
                         lst_color=None, 
                         save_path = os.path.join(save_path, 'Bootstrapping_Prob', title[:-11]), 
                         fwr_correction = False)
    
        plt.savefig(os.path.join(save_path, 'Bootstrapping_Prob' , 'Bootstrapping_' + title[:-11] + '.svg'))
        plt.close() 
       
    # =============================================================================
    # Critical Files Evaluation - Probability Score
    # =============================================================================
    df_critical_files = df.loc[df.audio.isin(df_critical.audio.unique()),:]
    # =============================================================================
    # Bootstrapping all Critical Files
    # =============================================================================
    pb.plot_boxplots(sorted(list(df_critical.audio.unique())),
                      df_critical_files,
                     'audio',
                     ['negative', 'neutral', 'positive'], 
                     boot = 'mean', 
                     boot_size = 5000,
                     title='All Responses for Critical Audiofiles', 
                     lst_color=None, 
                     save_path = os.path.join(save_path, 'Bootstrapping_Prob', 'All_Critical'), 
                     fwr_correction = False)

    plt.savefig(os.path.join(save_path, 'Bootstrapping_Prob' , 'Bootstrapping_all_critical.svg'))
    plt.close()
    
    # =============================================================================
    # Evaluation
    # =============================================================================        

    for cond in list(df_critical_files.condition.unique()):
        df_plot = df_critical_files.loc[df_critical_files.condition == cond,['audio','negative', 'neutral', 'positive']].melt(id_vars =['audio'])

        df_plot['variable'] = [row[:8] for row in df_plot['variable']]
        pb.plot_eva_audio(sorted(list(df_plot['variable'].unique())),
                         df_plot,
                         'variable',
                         'audio',
                         sorted(list(df_plot['audio'].unique())), 
                         boot = 'mean', 
                         boot_size = 5000,
                         title='Evaluation of Audiosegments', 
                         lst_color=None, 
                         save_path = os.path.join(save_path, 'Evaluation'), 
                         fwr_correction = False)
    
        plt.savefig(os.path.join(save_path, 'Evaluation' , 'Evaluation_' + cond + '.svg'))
        plt.close()    


