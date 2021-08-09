# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import scipy.io
import scipy.signal
from sklearn.model_selection import train_test_split
import pickle


ds_ECG, ds_EEG, ds_GSR = {}, {}, {}
arousal, valence = [], []
excluded_subjects = [44, 52]
# Si escludono soggetti i cui dati sono imprecisi o incompleti

# Caricamento soggetto per soggetto
for i in range(37, 56):
    if(i not in excluded_subjects):

        # Caricamento dati
        clips_ECG = {}
        for j in range(1, 37):
          clips_ECG[('n_' + str(j))] = scipy.io.loadmat('ECGData/Movie_P' + str(i) + '/ECG_Clip' + str(j))['Data_ECG'][:,1:]
        ds_ECG[('S' + str(i))] = clips_ECG
        arousal.append(scipy.io.loadmat('Dt_SelfReports.mat')['Ratings'][0][i])
        valence.append(scipy.io.loadmat('Dt_SelfReports.mat')['Ratings'][1][i])
        clips_EEG = {}
        for j in range(1, 37):
          clips_EEG[('n_' + str(j))] = np.transpose(scipy.io.loadmat('EEGData/Movie_P' + str(i) + '/EEG_Clip' + str(j))['ThisEEG'][:,1:])
        ds_EEG[('S' + str(i))] = clips_EEG
        clips_GSR = {}
        for j in range(1, 37):
          clips_GSR[('n_' + str(j))] = scipy.io.loadmat('GSRData/Movie_P' + str(i) + '/GSR_Clip' + str(j))['Data_GSR'][:,1:]
        ds_GSR[('S' + str(i))] = clips_GSR
print("Subjects loaded")

# Creazione delle 4 classi:
#   0 = arousal > 3 & valence > 0
#   1 = arousal > 3 & valence <= 0
#   2 = arousal <= 3 & valence > 0
#   3 = arousal <= 3 & valence <= 0
Y = []
for i in range(0, len(arousal)):
    for j in range(0, len(arousal[i])):
        if(arousal[i][j] > 3):
            if(valence[i][j] > 0):
                Y.append(0)
            elif(valence[i][j] <= 0):
                Y.append(1)
        elif(arousal[i][j] <= 3):
            if(valence[i][j] > 0):
                Y.append(2)
            elif(valence[i][j] <= 0):
                Y.append(3)
print("Targets loaded")

# Rimozione dei valori incompleti
for s in ds_ECG.keys():
    for clip in ds_ECG[s].keys():
        ds_ECG[s][clip] = ds_ECG[s][clip][~np.isnan(ds_ECG[s][clip]).any(axis = 1)]
        ds_EEG[s][clip] = ds_EEG[s][clip][~np.isnan(ds_EEG[s][clip]).any(axis = 1)]
        ds_GSR[s][clip] = ds_GSR[s][clip][~np.isnan(ds_GSR[s][clip]).any(axis = 1)]
print("Removed nan from data")

# Ricampionamento a 32 Hz
for s in ds_ECG.keys():
    for clip in ds_ECG[s].keys():
        ds_ECG[s][clip] = scipy.signal.resample(ds_ECG[s][clip], len(ds_EEG[s][clip]))
        ds_GSR[s][clip]= scipy.signal.resample(ds_GSR[s][clip], len(ds_EEG[s][clip]))
print("Data resampled")

# Concatenamento delle features di ASCERTAIN
ds = {}
i = 0
for s in ds_ECG.keys():
  clips = {}
  j = 0
  for clip in ds_ECG[s].keys():
      clips['Clip' + str(j)] = np.concatenate([
      ds_ECG[s][clip],
      ds_EEG[s][clip],
      ds_GSR[s][clip],
      ], axis = 1)
      j += 1
  ds['S' + str(i)] = clips
  i += 1

Y = (np.array(Y, dtype = np.float32)).reshape(-1, 36)

# Creazione delle sottosequenze lunghe 160 (5 secondi)
SubsequencesX, SubsequencesY = [], []
# Si estrae mantenendo le classi bilanciate
count_0, count_1, count_2, count_3 = 0, 0, 0, 0
i = 0
for s in ds.keys():
    j = 0
    for clip in ds[s].keys():
        if(((Y[i][j] == 0) & (count_0 < 84)) or ((Y[i][j] == 1) & (count_1 < 84)) or ((Y[i][j] == 2) & (count_2 < 84)) or ((Y[i][j] == 3) & (count_3 < 84))):
           length = len(ds[s][clip])
           for k in range(0, 10):
               SubsequencesX.append(ds[s][clip][length-(160*(k+1)):length-(160*k),:])
               if (Y[i][j] == 0):
                    SubsequencesY.append(0)
               elif (Y[i][j] == 1):
                    count_1 += 1
                    SubsequencesY.append(1)
               elif (Y[i][j] == 2):
                    count_2 += 1
                    SubsequencesY.append(2)
               elif (Y[i][j] == 3):
                    count_3 += 1
                    SubsequencesY.append(3)
        j += 1
    i += 1

X_ASC = (np.array(SubsequencesX, dtype = np.float64)).reshape(-1, 160, 17)
Y = np.array(SubsequencesY, dtype = np.float32)
y_ASC = tf.keras.utils.to_categorical(Y, num_classes = 4)

# Divisione in training set e in test set
Xtr, Xts, ytr, yts = train_test_split(X_ASC, y_ASC, test_size = 0.25, train_size = 0.75, random_state=42)

with open("splitted/Xts.pkl", 'wb') as handle:
    pickle.dump(Xts, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("splitted/yts.pkl", 'wb') as handle:
    pickle.dump(yts, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Creazione dei soggetti fittizi a partire dal training set
n = 0
for S in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]:
    Xs = Xtr[108*n:108*(n+1)]
    ys = ytr[108*n:108*(n+1)]
    print("Shape of X:", Xs.shape)
    print("Shape of y:", ys.shape)
    n += 1

    with open("splitted/X" + S + ".pkl", 'wb') as handle:
        pickle.dump(Xs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("splitted/y" + S + ".pkl", 'wb') as handle:
        pickle.dump(ys, handle, protocol=pickle.HIGHEST_PROTOCOL)
