import numpy as np
import tensorflow as tf
import scipy.io
import scipy.signal
import pickle
from collections import Counter

ds_ECG, ds_EEG, ds_GSR, ds_Y = {}, {}, {}, {}
#Removing subjects with missing data
excluded_subjects = [44, 52]
#Excluding subjects data with poor accuracy & many nan values

print("Loading subjects...", end=" ")
for i in range(37, 57):
    if(i not in excluded_subjects):
        arousal = scipy.io.loadmat('/home/fexed/ML/datasets/ASCERTAIN/Dt_SelfReports.mat')['Ratings'][0][i]
        valence = scipy.io.loadmat('/home/fexed/ML/datasets/ASCERTAIN/Dt_SelfReports.mat')['Ratings'][1][i]
        clips_ECG = {}
        for j in range(1, 37):
            clips_ECG[('n_' + str(j))] = scipy.io.loadmat('/home/fexed/ML/datasets/ASCERTAIN/ECGData/Movie_P' + str(i) + '/ECG_Clip' + str(j))['Data_ECG'][:,1:]
        ds_ECG[('S' + str(i))] = clips_ECG
        clips_EEG = {}
        for j in range(1, 37):
            clips_EEG[('n_' + str(j))] = np.transpose(scipy.io.loadmat('/home/fexed/ML/datasets/ASCERTAIN/EEGData/Movie_P' + str(i) + '/EEG_Clip' + str(j))['ThisEEG'][:,1:])
        ds_EEG[('S' + str(i))] = clips_EEG
        clips_GSR = {}
        for j in range(1, 37):
            clips_GSR[('n_' + str(j))] = scipy.io.loadmat('/home/fexed/ML/datasets/ASCERTAIN/GSRData/Movie_P' + str(i) + '/GSR_Clip' + str(j))['Data_GSR'][:,1:]
        ds_GSR[('S' + str(i))] = clips_GSR

        #4 classes:
        #0 = arousal > 3 & valence > 0
        #1 = arousal > 3 & valence <= 0
        #2 = arousal <= 3 & valence > 0
        #3 = arousal <= 3 & valence <= 0
        ds_Y[('S' + str(i))] = []
        for j in range(0, len(arousal)):
            if(arousal[j] > 3):
                if(valence[j]> 0):
                    ds_Y[('S' + str(i))].append(0)
                elif(valence[j] <= 0):
                    ds_Y[('S' + str(i))].append(1)
            elif(arousal[j] <= 3):
                if(valence[j] > 0):
                    ds_Y[('S' + str(i))].append(2)
                elif(valence[j] <= 0):
                    ds_Y[('S' + str(i))].append(3)
        #print("S" + str(i) + " : " + str(len(ds_Y[('S' + str(i))])) + " " + str(len(ds_GSR[('S' + str(i))])) + " " + str(len(ds_EEG[('S' + str(i))])) + " " + str(len(ds_ECG[('S' + str(i))])))
print("done")


print("Removing NaNs...", end=" ")
#Removing rows containing nan values
for s in ds_ECG.keys():
    for clip in ds_ECG[s].keys():
        ds_ECG[s][clip] = ds_ECG[s][clip][~np.isnan(ds_ECG[s][clip]).any(axis = 1)]
        ds_EEG[s][clip] = ds_EEG[s][clip][~np.isnan(ds_EEG[s][clip]).any(axis = 1)]
        ds_GSR[s][clip] = ds_GSR[s][clip][~np.isnan(ds_GSR[s][clip]).any(axis = 1)]
print("done")

print("Resampling...", end=" ")
#Resampling data to lower freq, 32 Hz
for s in ds_ECG.keys():
    for clip in ds_ECG[s].keys():
        ds_ECG[s][clip] = scipy.signal.resample(ds_ECG[s][clip], len(ds_EEG[s][clip]))
        ds_GSR[s][clip]= scipy.signal.resample(ds_GSR[s][clip], len(ds_EEG[s][clip]))
print("done")

print("Merging...", end=" ")
#Merging features
ds, dsY = {}, {}
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
    dsY['S' + str(i)] = ds_Y[s]
    #print(s + " -> S" + str(i) + " : " + str(len(ds['S' + str(i)])) + " (" + str(len(ds['S' + str(i)]["Clip0"])) + ") (" + str(len(ds['S' + str(i)]["Clip0"][0])) + ") " + str(len(dsY['S' + str(i)])))
    i += 1
print("done")

print("Saving...")
#Creating X and Y list of subsequences, length = 160 (3 seconds)
#Avoiding sampling bias
for s in ds.keys():
    print(s, end=" ")
    Y = dsY[s]
    SubsequencesX, SubsequencesY = [], []
    count_0, count_1, count_2, count_3 = 0, 0, 0, 0
    j = 0
    for clip in ds[s].keys():
       length = len(ds[s][clip])
       k = 0
       while (length - (160*(k+1))) > 0:
           #print(str(Y[j]) + " - " + str(count_0) + " " + str(count_1) + " " + str(count_2) + " " + str(count_3))
           if ((Y[j] == 0)  & (count_0 < 84)):
                count_0 += 1
                SubsequencesX.append(ds[s][clip][length-(160*(k+1)):length-(160*k),:])
                SubsequencesY.append(0)
           elif ((Y[j] == 1) & (count_1 < 84)):
                count_1 += 1
                SubsequencesX.append(ds[s][clip][length-(160*(k+1)):length-(160*k),:])
                SubsequencesY.append(1)
           elif ((Y[j] == 2) & (count_2 < 84)):
                count_2 += 1
                SubsequencesX.append(ds[s][clip][length-(160*(k+1)):length-(160*k),:])
                SubsequencesY.append(2)
           elif ((Y[j] == 3) & (count_3 < 84)):
                count_3 += 1
                SubsequencesX.append(ds[s][clip][length-(160*(k+1)):length-(160*k),:])
                SubsequencesY.append(3)
           k += 1
       j += 1

    #print(Counter(SubsequencesY))
    Xs = (np.array(SubsequencesX, dtype = np.float64)).reshape(-1, 160, 17)
    ys = tf.keras.utils.to_categorical(np.array(SubsequencesY, dtype = np.float32), num_classes = 4)
    print(Xs.shape)
    #print("Shape of X" + s + ":", Xs.shape)
    #print("Shape of y" + s + ":", ys.shape)

    with open("/home/fexed/ML/datasets/ASCERTAIN/splitted/X" + s + ".pkl", 'wb') as handle:
        pickle.dump(Xs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("/home/fexed/ML/datasets/ASCERTAIN/splitted/y" + s + ".pkl", 'wb') as handle:
        pickle.dump(ys, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("done")
