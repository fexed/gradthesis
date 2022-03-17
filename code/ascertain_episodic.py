import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import copy
import random


# Log directory
def get_run_logdir(label=""):
    import time
    run_id = time.strftime("continual_ASCERTAIN_%Y%m%d_%H%M%S")
    return os.path.join("/home/fexed/ML/tensorboard_logs", run_id + "_" + label)


print(os.getpid())
input("Press Enter to continue...")
print("Opening test set")
Xts = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/Xts.pkl", 'rb'), encoding='latin1')
yts = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/yts.pkl", 'rb'), encoding='latin1')
Xts_1, Xts_2, Xts_3, Xts_4, yts_1, yts_2, yts_3, yts_4 = [], [], [], [], [], [], [], []
idx_1 = [i for i, x in enumerate(yts) if x[0] == 1]
for i in idx_1:
    Xts_1.append(Xts[i])
    yts_1.append([1., 0., 0., 0.])
Xts_1, yts_1 = np.array(Xts_1), np.array(yts_1)
idx_2 = [i for i, x in enumerate(yts) if x[1] == 1]
for i in idx_2:
    Xts_2.append(Xts[i])
    yts_2.append([0., 1., 0., 0.])
Xts_2, yts_2 = np.array(Xts_2), np.array(yts_2)
idx_3 = [i for i, x in enumerate(yts) if x[2] == 1]
for i in idx_3:
    Xts_3.append(Xts[i])
    yts_3.append([0., 0., 1., 0.])
Xts_3, yts_3 = np.array(Xts_3), np.array(yts_3)
idx_4 = [i for i, x in enumerate(yts) if x[3] == 1]
for i in idx_4:
    Xts_4.append(Xts[i])
    yts_4.append([0., 0., 0., 1.])
Xts_4, yts_4 = np.array(Xts_4), np.array(yts_4)
print("and splitted in " + str(len(Xts_1)) + " + " + str(len(Xts_2)) + " + " + str(len(Xts_3)) + " + " + str(len(Xts_4)) + " datapoints")

print("Creating model")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.GRU(24, return_sequences = True, input_shape=(160, 17)))
model.add(tf.keras.layers.GRU(24))
model.add(tf.keras.layers.Dense(4, activation = 'softmax',
                  kernel_regularizer = tf.keras.regularizers.l1_l2(l1 = 1e-5, l2 = 1e-5),
                  bias_regularizer = tf.keras.regularizers.l1_l2(l1 = 1e-5, l2 = 1e-5),
                  activity_regularizer = tf.keras.regularizers.l1_l2(l1 = 1e-5, l2 = 1e-5)))
opt = tf.keras.optimizers.Adam(learning_rate = 0.01, beta_1 = 0.9, beta_2 = 0.99)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

R = []
T = 4
E = 8
b = [model.evaluate(Xts_1, yts_1, verbose = 0)[1], model.evaluate(Xts_2, yts_2, verbose = 0)[1], model.evaluate(Xts_3, yts_3, verbose = 0)[1], model.evaluate(Xts_4, yts_4, verbose = 0)[1]]
ACC, BWT, FWT = 0, 0, 0
modeltime = 0
epochs = []
scores = []
X, y = None, None
X_1, X_2, X_3, X_4, y_1, y_2, y_3, y_4 = [], [], [], [], [], [], [], []  # memorie task
m = 70

print("Training\tstarted")
for S in [("S0", "S1"), ("S2", "S3"), ("S4", "S5"), ("S6", "S7"), ("S8", "S9"), ("S10", "S11"), ("S12", "S13"), ("S14", "S15")]:
    print("Subjects\t" + S[0] + " " + S[1], end="\r")
    Xa = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/X" + S[0] + ".pkl", 'rb'), encoding='latin1')
    ya = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/y" + S[0] + ".pkl", 'rb'), encoding='latin1')
    Xb = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/X" + S[1] + ".pkl", 'rb'), encoding='latin1')
    yb = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/y" + S[1] + ".pkl", 'rb'), encoding='latin1')

    if (X is None):
        X = np.concatenate([Xa, Xb], axis = 0)
        y = np.concatenate([ya, yb], axis = 0)
    else:
        X = np.concatenate([X, Xa, Xb], axis = 0)
        y = np.concatenate([y, ya, yb], axis = 0)
    del Xa
    del Xb
    del ya
    del yb

    Xtr, Xvl, ytr, yvl = train_test_split(X, y, test_size = 0.1, train_size = 0.9, random_state=42)
    logdir = get_run_logdir(S[0] + S[1] + "_cumulative")
    es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, verbose = 0, restore_best_weights = True)
    tb = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)
    start = time.time()
    graph = model.fit(Xtr, ytr, epochs = 100, validation_data = (Xvl, yvl), callbacks = [es, tb], verbose = 0, batch_size=256, use_multiprocessing=True, workers=8)
    end = time.time()
    modeltime += (end - start)
    history = model.history.history['val_loss']
    epochs.append(np.argmin(history) + 1)
    scores.append(model.evaluate(Xts, yts, verbose = 0)[1])

    R.append([model.evaluate(Xts_1, yts_1, verbose = 0)[1], model.evaluate(Xts_2, yts_2, verbose = 0)[1], model.evaluate(Xts_3, yts_3, verbose = 0)[1], model.evaluate(Xts_4, yts_4, verbose = 0)[1]])

    #X, _, y, _ = train_test_split(Xtr, ytr, test_size = 0.75, train_size = 0.25, random_state=42)  # replay random di elementi precedenti
    idx_1 = [i for i, x in enumerate(ytr) if x[0] == 1]
    for i in idx_1:
        X_1.append(Xtr[i])
        y_1.append([1., 0., 0., 0.])
    random.shuffle(X_1)
    X_1 = X_1[0:m]
    y_1 = y_1[0:m]
    try:
        X = np.concatenate([X_1], axis = 0)
        y = np.concatenate([y_1], axis = 0)
    except ValueError:
        X = []
        y = []

    idx_2 = [i for i, x in enumerate(ytr) if x[1] == 1]
    for i in idx_2:
        X_2.append(Xtr[i])
        y_2.append([0., 1., 0., 0.])
    random.shuffle(X_2)
    X_2 = X_2[0:m]
    y_2 = y_2[0:m]
    try:
        X = np.concatenate([X, X_2], axis = 0)
        y = np.concatenate([y, y_2], axis = 0)
    except ValueError:
        X = X
        y = y

    idx_3 = [i for i, x in enumerate(ytr) if x[2] == 1]
    for i in idx_3:
        X_3.append(Xtr[i])
        y_3.append([0., 0., 1., 0.])
    random.shuffle(X_3)
    X_3 = X_3[0:m]
    y_3 = y_3[0:m]
    try:
        X = np.concatenate([X, X_3], axis = 0)
        y = np.concatenate([y, y_3], axis = 0)
    except ValueError:
        X = X
        y = y

    idx_4 = [i for i, x in enumerate(ytr) if x[3] == 1]
    for i in idx_4:
        X_4.append(Xtr[i])
        y_4.append([0., 0., 0., 1.])
    random.shuffle(X_4)
    X_4 = X_4[0:m]
    y_4 = y_4[0:m]
    try:
        X = np.concatenate([X, X_4], axis = 0)
        y = np.concatenate([y, y_4], axis = 0)
    except ValueError:
        X = X
        y = y

print("Training\tended  ")
t = 0
for i in range(T):  # Accuratezza media
        t += R[E-1][i]
ACC = t/T

t = 0
for i in range(T):  # BWT
        m = 0
        for j in range(E):
            m += (R[E-1][i] - R[j][i])
        t += m/E
BWT = t/(T)

t = 0
for i in range(T):
        m = 0
        for j in range(E):
            m += (R[j][i] - b[i])
        t += m/E
FWT = t/(T)
print('Media numero epoche:', np.around(np.mean(epochs), 2))
print('Deviazione standard numero epoche:', np.around(np.std(epochs), 2))
print('Media tempo di addestramento:', np.around(modeltime/8, 2), 's')
print('Accuracy:', str(list(scores)), '%')
print('Media accuracy:', np.around(np.mean(scores), 4)*100, '%')
print('Deviazione standard accuracy:', np.around(np.std(scores), 4)*100, '%')
print('ACC\t', np.around(ACC, 4))
print('BWT\t', np.around(BWT, 4))
print('FWT\t', np.around(FWT, 4))
#plt.plot(scores)
#plt.savefig("accuracy_sum.png")
