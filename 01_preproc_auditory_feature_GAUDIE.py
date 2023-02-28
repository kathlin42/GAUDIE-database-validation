# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:21:53 2021

@author: hirning
"""
# =============================================================================
# Load Packages
# =============================================================================

import numpy as np
import os
from scipy.signal import fftconvolve
from scipy.special import rel_entr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import librosa
import soundfile as sf
from librosa import display
import pandas as pd
import seaborn as sns
import pickle
from scipy.stats.stats import pearsonr
import scipy.stats as stats
from pydub import AudioSegment
from helper_plotting import plot_boxes
import config_analysis
# =============================================================================
# Find max length in stimuli 
# =============================================================================
project_directory = os.path.join(config_analysis.project_root, "gaudie_audio_validation")

db = -20
hop_length = 512
fig_format = ['.svg', '.png']
data_path = os.path.join(project_directory, 'ressources', 'audio_stimuli', 'GAUDIE/preproc_audio/'+ str(db) +'db_normalized_audio')
save_path = os.path.join(project_directory, 'derivatives', 'encoding', 'encoding_auditory_feature_librosa')

df_describe = pd.read_csv(os.path.join(project_directory, 'ressources', 'audio_stimuli', "Description_GAUDIE_stimuli.csv"), sep=';', decimal = ',', header = 0)
max_files = max(df_describe['Condition'].value_counts())

max_length = np.round(max(df_describe['Samples'])/df_describe['Sample_rate'].unique()[0])
min_length = np.round(min(df_describe['Samples'])/df_describe['Sample_rate'].unique()[0])
# =============================================================================
# Process and Analyse Auditory Feature of Stimuli with librosa
# =============================================================================
i_count = 0
for i_condition, condition in enumerate([condition for condition in os.listdir(data_path) if condition != 'test']):
    save_cond = os.path.join(save_path, condition)
    if not os.path.exists(save_cond):
        os.makedirs(save_cond)
    
    for i, sound_file in enumerate(sorted(os.listdir(os.path.join(data_path, condition)))):
        i = sound_file[-6:-4]
        print(sound_file, i)
        y, sr = librosa.load(os.path.join(data_path, condition, sound_file))
        
        #Create Stimuli of same length
        while (y.shape[0] / sr) < max_length:
            y = np.append(y, y)   
            y = y[:int(max_length*sr)]
        # =============================================================================
        # Within_Condition Similarity
        # =============================================================================
        if i_condition == 0: 
            within_similiarity_crosscorr = np.zeros((len([condition for condition in os.listdir(data_path) if condition != 'test']), max_files * max_files, librosa.stft(y).shape[1]))
            within_similiarity_crosscorr[:,:,:] = np.nan
            within_similiarity_kl_div = np.zeros((len([condition for condition in os.listdir(data_path) if condition != 'test']), max_files * max_files, librosa.stft(y).shape[1]))
            within_similiarity_kl_div[:,:,:] = np.nan
            within_similiarity_kl_div_reverse = np.zeros((len([condition for condition in os.listdir(data_path) if condition != 'test']), max_files * max_files, librosa.stft(y).shape[1]))
            within_similiarity_kl_div_reverse[:,:,:] = np.nan
            within_similiarity_kl_div_average = np.zeros((len([condition for condition in os.listdir(data_path) if condition != 'test']), max_files * max_files, librosa.stft(y).shape[1]))
            within_similiarity_kl_div_average[:,:,:] = np.nan
            rms_average = np.zeros((len([condition for condition in os.listdir(data_path) if condition != 'test']),max_files, librosa.stft(y).shape[1]))
            within_similiarity_kl_div_average[:,:,:] = np.nan
            
        df_within_similiarity_crosscorr = pd.DataFrame(columns=sorted(os.listdir(os.path.join(data_path, condition))))
        df_within_similiarity_kl_div = pd.DataFrame(columns=sorted(os.listdir(os.path.join(data_path, condition))))
        df_within_similiarity_crosscorr['Compared Audio Segments'] = sorted(os.listdir(os.path.join(data_path, condition)))
        df_within_similiarity_kl_div['Compared Audio Segments'] = sorted(os.listdir(os.path.join(data_path, condition)))

        
        if condition == [condition for condition in os.listdir(data_path) if condition != 'test'][0]:
            rms_cond_1 = []
            names_cond_1 = []           
        elif condition == [condition for condition in os.listdir(data_path) if condition != 'test'][1]:       
            rms_cond_2 = []
            names_cond_2 = []      
        elif condition == [condition for condition in os.listdir(data_path) if condition != 'test'][2]:       
            rms_cond_3 = []
            names_cond_3 = [] 
                    
                        
        # =============================================================================
        # Compute the RMS     
        # =============================================================================
        '''
        Compute root-mean-square (RMS) value for each frame, either from the audio samples y or from a spectrogram S.
        Computing the RMS value from audio samples is faster as it doesn’t require a STFT calculation. 
        However, using a spectrogram will give a more accurate representation of energy over time because its frames can be windowed, 
        thus prefer using S if it’s already available.
        '''

        S, phase = librosa.magphase(librosa.stft(y))       
        rms = librosa.feature.rms(S=S)
        
        fig, ax = plt.subplots(nrows=2, sharex=True)
        times = librosa.times_like(rms)
        ax[0].semilogy(times, rms[0], label='RMS Energy')
        ax[0].set(xticks=[])
        ax[0].legend()
        ax[0].label_outer()
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                 y_axis='log', x_axis='time', ax=ax[1])
        ax[1].set(title='log Power spectrogram')
        fig.suptitle('RMS from spectrogram - ' + sound_file[17:-4].replace('_', ' '))
        fig.tight_layout()
        for end in fig_format:
            plt.savefig(os.path.join(save_cond, 'RMS_spectrogram_' + sound_file[:-4] + end))
        plt.close()
    
    
        # =============================================================================
        # Melspectrogram
        # =============================================================================
        '''
        Compute a mel-scaled spectrogram.
        If a spectrogram input S is provided, then it is mapped directly onto the mel basis by mel_f.dot(S).
        If a time-series input y, sr is provided, then its magnitude spectrogram S is first computed, and then mapped onto the mel scale by mel_f.dot(S**power).
        '''
        
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram - ' + sound_file[17:-4].replace('_', ' '))
        for end in fig_format:
            plt.savefig(os.path.join(save_cond, 'Mel-frequency_spectrogram_' + sound_file[:-4] + end))
        plt.close()
        # =============================================================================
        # Compute a chromagram from a waveform or power spectrogram. 
        # Identifying different pitch classes
        # =============================================================================
        S = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=S, sr=sr)
        
        fig, ax = plt.subplots()
        img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax)
        fig.suptitle('Chromagram - ' + sound_file[17:-4].replace('_', ' '))
        fig.tight_layout()
        for end in fig_format:
            plt.savefig(os.path.join(save_cond, 'Chromagram_' + sound_file[:-4] + end))
        plt.close()
        # =============================================================================
        # Decomposition of the spectrogram
        # =============================================================================
        '''
        Decompose a feature matrix.
        Given a spectrogram S, produce a decomposition into components and activations such that S ~= components.dot(activations).
        By default, this is done with with non-negative matrix factorization (NMF), but any sklearn.decomposition-type object will work.
        '''

        comps, acts = librosa.decompose.decompose(S, n_components=16, sort=True)
        fig, ax = plt.subplots(nrows=1, ncols=2)
        librosa.display.specshow(librosa.amplitude_to_db(comps,ref=np.max),y_axis='log', ax=ax[0])
        ax[0].set(title='Components')
        librosa.display.specshow(acts, x_axis='time', ax=ax[1])
        ax[1].set(ylabel='Components', title='Activations')
        fig.suptitle('Spectrogram decomposition - ' + sound_file[17:-4].replace('_', ' '))
        fig.tight_layout()
        for end in fig_format:
            plt.savefig(os.path.join(save_cond, 'Spectrogram_decomposition_' + sound_file[:-4] + end))
        plt.close()
        # =============================================================================
        # Tempogram 
        # =============================================================================
            
        # Compute local onset autocorrelation
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
        # Compute global onset autocorrelation
        ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
        ac_global = librosa.util.normalize(ac_global)

        # Estimate the global tempo for display purposes
        tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)[0]

        # Plot 
        fig, ax = plt.subplots(nrows=4, figsize=(10, 10))
        times = librosa.times_like(oenv, sr=sr, hop_length=hop_length)
        ax[0].plot(times, oenv, label='Onset strength')
        ax[0].label_outer()
        ax[0].legend(frameon=True)
        librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='tempo', cmap='magma',
                         ax=ax[1])

        ax[1].axhline(tempo, color='w', linestyle='--', alpha=1,
            label='Estimated tempo={:g}'.format(tempo))
        ax[1].legend(loc='upper right')
        ax[1].set(title='Tempogram')
        x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr, num=tempogram.shape[0])
        ax[2].plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
        ax[2].plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
        ax[2].set(xlabel='Lag (seconds)')
        ax[2].legend(frameon=True)
        
        freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
        ax[3].semilogx(freqs[1:], np.mean(tempogram[1:], axis=1), label='Mean local autocorrelation', basex=2)
        ax[3].semilogx(freqs[1:], ac_global[1:], '--', alpha=0.75, label='Global autocorrelation', basex=2)
        ax[3].axvline(tempo, color='black', linestyle='--', alpha=.8,label='Estimated tempo={:g}'.format(tempo))
        ax[3].legend(frameon=True)
        ax[3].set(xlabel='BPM')
        ax[3].grid(True)
        fig.suptitle(sound_file[17:-4].replace('_', ' ').upper())
        fig.tight_layout()
        for end in fig_format:
            plt.savefig(os.path.join(save_cond, 'Tempogram_' + sound_file[:-4] + end))
        plt.close()
   
        # =============================================================================
        # Compute Tempogram with Fourier Analysis
        # =============================================================================
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
        
        # Compute the auto-correlation tempogram, unnormalized to make comparison easier
        ac_tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length, norm=None)
        
        fig, ax = plt.subplots(nrows=3, sharex=True)
        ax[0].plot(librosa.times_like(oenv), oenv, label='Onset strength')

        ax[0].legend(frameon=True)
        ax[0].label_outer()
        librosa.display.specshow(np.abs(tempogram), sr=sr, hop_length=hop_length, x_axis='time', y_axis='fourier_tempo', cmap='magma', ax=ax[1])
        ax[1].set(title='Fourier tempogram')
        ax[1].label_outer()
        librosa.display.specshow(ac_tempogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='tempo', cmap='magma',
                                 ax=ax[2])
        ax[2].set(title='Autocorrelation tempogram')
        fig.suptitle(sound_file[17:-4].replace('_', ' ').upper())
        fig.tight_layout()
        for end in fig_format:
            plt.savefig(os.path.join(save_cond, 'Tempogram_Fourier_' + sound_file[:-4] + end))
        plt.close()
        # =============================================================================
        # Compare similarity
        # =============================================================================
           
        # =============================================================================
        # Within group  -  Aggregate with max(abs())
        # =============================================================================
  
        if condition == [condition for condition in os.listdir(data_path) if condition != 'test'][0]:
            rms_cond_1.append(rms.mean(-1)[0])
            names_cond_1.append(sound_file[17:-4])   
        elif condition == [condition for condition in os.listdir(data_path) if condition != 'test'][1]:
            rms_cond_2.append(rms.mean(-1)[0])
            names_cond_2.append(sound_file[17:-4])  
        elif condition == [condition for condition in os.listdir(data_path) if condition != 'test'][2]:
            rms_cond_3.append(rms.mean(-1)[0])
            names_cond_3.append(sound_file[17:-4])   

        # =============================================================================
        # Within group  -  Aggregate with max(abs())
        # =============================================================================
      
        rms_average[i_condition,i,:] = rms
        for i_temp_file, temp_file in enumerate(sorted(os.listdir(os.path.join(data_path, condition)))):
            print(temp_file)
            y_temp, sr_temp = librosa.load(os.path.join(data_path, condition, temp_file))
            S_temp, phase_temp = librosa.magphase(librosa.stft(y_temp))       
            rms_temp = librosa.feature.rms(S=S_temp)
            
            within_similiarity_crosscorr[i_condition,i_count,:] = fftconvolve(rms[0,:], rms_temp[0,:], mode='same')
            within_similiarity_kl_div[i_condition,i_count,:] = rel_entr(rms[0,:], rms_temp[0,:])
                        
            df_within_similiarity_crosscorr.loc[df_within_similiarity_crosscorr['Compared Audio Segments'] ==sound_file[17:-4], temp_file[17:-4]] = np.mean(abs(fftconvolve(rms[0,:], rms_temp[0,:], mode='same')))
            df_within_similiarity_kl_div.loc[df_within_similiarity_crosscorr['Compared Audio Segments'] ==sound_file[17:-4], temp_file[17:-4]] = np.mean(abs(rel_entr(rms[0,:], rms_temp[0,:])))
                               
            i_count = i_count + 1    
    

    df_within_similiarity_crosscorr = df_within_similiarity_crosscorr.set_index('Compared Audio Segments')
    df_within_similiarity_crosscorr = df_within_similiarity_crosscorr.fillna(0)

    df_within_similiarity_kl_div = df_within_similiarity_kl_div.set_index('Compared Audio Segments')
    df_within_similiarity_kl_div = df_within_similiarity_kl_div.fillna(0)
    

    fig = plt.figure(figsize=(22, 10))
    axs = fig.subplots(1,2, sharey='all').flatten()
    fig.suptitle('Similarity Analysis within ' + condition.capitalize() + ' Condition', fontsize = 20)
    
    for ax, (name, measure) in enumerate([('Averaged Cross-Correlation', df_within_similiarity_crosscorr), ('Averaged Realtive Entropy', df_within_similiarity_kl_div)]):
        measure = measure.replace([np.inf, -np.inf], np.nan)
        measure = measure.fillna(0)
        sns.heatmap(measure, ax = axs[ax])
        axs[ax].set_title(name, fontsize = 14)    
    fig.tight_layout( rect=[0, 0.1, 1.0, 0.99])
    
    for end in fig_format:
        plt.savefig(os.path.join(save_cond,  'Heatmap_similiarity_crosscorr_relentropy' + end))
    plt.close()
    df_within_similiarity_crosscorr.to_csv(os.path.join(save_cond,"df_within_similiarity_crosscorr.csv"), sep=';', decimal = ',', index=True, header=True) 
    df_within_similiarity_kl_div.to_csv(os.path.join(save_cond,"df_within_similiarity_kl_div.csv"), sep=';', decimal = ',', index=True, header=True)                      
        

with open(os.path.join(save_path, 'within_similiarity_crosscorr.pickle'), 'wb') as file:
    pickle.dump(within_similiarity_crosscorr, file, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(save_path, 'within_similiarity_kl_div.pickle'), 'wb') as file:
    pickle.dump(within_similiarity_kl_div, file, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(os.path.join(save_path, 'rms_average.pickle'), 'wb') as file:
    pickle.dump(rms_average, file, protocol=pickle.HIGHEST_PROTOCOL)

# =============================================================================
# Between Group Comparison 
# =============================================================================

rms_grand_average = np.nanmean(rms_average, axis = 1)
rms_grand_average_crosscorr = fftconvolve(rms_grand_average[0,:], rms_grand_average[1,:], mode='same')
rms_grand_average_rel_entr = rel_entr(rms_grand_average[0,:], rms_grand_average[1,:])
timevector = np.linspace(0, int(max_length), num=rms_grand_average.shape[-1])
lst_color=['#1F82C0', '#E2001A', '#B1C800', '#179C7D', '#F29400', 'darkviolet']
f, (ax1, ax2) = plt.subplots(2,1, figsize=(20,8), sharex=True, sharey=False)
legend = []
for i, (name, measure) in enumerate([('Averaged Cross-Correlation', rms_grand_average_crosscorr), ('Averaged Realtive Entropy', rms_grand_average_rel_entr)]):
    legend.append(name)
    if i == 0:
        ax1.plot(timevector, measure, color = lst_color[i])
        ax1.set_title(name, fontsize = 14)    
    else:
        ax2.plot(timevector, measure, color = lst_color[i])
        ax2.set_title('Averaged Realtive Entropy', fontsize = 14) 
f.suptitle('Similarity Analysis between Conditions Averaged over Audio Segments', fontsize = 20)      
f.legend([Line2D([0],[0], color=lst_color[0], lw=3),
          Line2D([0], [0], color=lst_color[1], lw=3)],
          legend,
          fontsize='medium')
for end in fig_format:
    plt.savefig(os.path.join(save_path, 'Similarity_analysis_between_conditions' + end))
plt.close()


# =============================================================================
# Average RMS per Condition
# =============================================================================
conditions = [condition for condition in os.listdir(data_path) if condition != 'test']
names = [names_cond_1, names_cond_2, names_cond_3]
fig = plt.figure(figsize=(18, 20))
axs = fig.subplots(1,len(names), sharey='all').flatten()
fig.suptitle('Similarity Analysis via the RMS', fontsize = 20)

for ax, (name, measure) in enumerate([('Average RMS Condition ' + [condition for condition in os.listdir(data_path) if condition != 'test'][0].capitalize(), rms_cond_1), 
                                      ('Average RMS Condition ' + [condition for condition in os.listdir(data_path) if condition != 'test'][1].capitalize(), rms_cond_2),
                                      ('Average RMS Condition ' + [condition for condition in os.listdir(data_path) if condition != 'test'][2].capitalize(), rms_cond_3)]):
    
    axs[ax].bar(names[ax], measure)
    axs[ax].set_title(name, fontsize = 14)    
    fig.tight_layout( rect=[0, 0.1, 1.0, 0.99])
   
for end in fig_format:
    plt.savefig(os.path.join(save_path,  'Barplot_per_condition' + end))
plt.close()

# =============================================================================
# Between Group Comparison 
# =============================================================================

df_combined = pd.DataFrame(rms_cond_1 + rms_cond_2 + rms_cond_3, columns=['Average RMS'])
df_combined['Condition'] = list(len(names_cond_1) * [conditions[0]]) + list(len(names_cond_2) * [conditions[1]]) + list(len(names_cond_3) * [conditions[2]])

df_CI = plot_boxes([conditions[0], conditions[1], conditions[2]],df_combined,'Condition','Average RMS', boot = 'mean',  
               boot_size = 5000,title= 'Comparison of the Average RMS per Condition', ax = None,lst_color=['#1F82C0', '#E2001A'], alpha = 0.95)
for end in fig_format:
    plt.savefig(os.path.join(save_path, 'Bootstrapped_CI' + end))
plt.close()
df_CI['Condition'] = conditions
df_CI = df_CI.set_index('Condition')
df_CI.to_csv(os.path.join(save_path, "Bootstrapped_CI.csv", sep=';', decimal = ',', index=True, header=True))       