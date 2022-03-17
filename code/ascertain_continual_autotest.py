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
    run_id = time.strftime("autotest_ASCERTAIN_%Y%m%d%H%M%S")
    return os.path.join("/home/fexed/ML/tensorboard_logs", run_id + "_" + label)


def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.GRU(24, return_sequences=True, input_shape=(160, 17)))
    model.add(tf.keras.layers.GRU(24))
    model.add(tf.keras.layers.Dense(4, activation = 'softmax',
                      kernel_regularizer = tf.keras.regularizers.l1_l2(l1 = 1e-6, l2 = 1e-6),
                      bias_regularizer = tf.keras.regularizers.l1_l2(l1 = 1e-4, l2 = 1e-4),
                      activity_regularizer = tf.keras.regularizers.l1_l2(l1 = 1e-4, l2 = 1e-4)))
    opt = tf.keras.optimizers.Adam(learning_rate = 0.005, beta_1 = 0.85, beta_2 = 0.999)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    return model


print(os.getpid())
input("Press Enter to continue...")
fig, ax = plt.subplots()
print("ASCERTAIN...", end=" ")
DSX, DSy = {}, {}
for S in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]:
    DSX[S] = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/X" + S + ".pkl", 'rb'), encoding='latin1')
    DSy[S] = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/y" + S + ".pkl", 'rb'), encoding='latin1')
print("loaded")
print("Test set...", end=" ")
Xts = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/XS17.pkl", 'rb'), encoding='latin1')
yts = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/yS17.pkl", 'rb'), encoding='latin1')
print("loaded (" + str(len(Xts)) + " datapoints)", end=" ")
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

print("\n\nTOTAL TRAINING")
print("Model...", end=" ")
model = create_model()
print("created")
modeltime = 0
epochs = []
scores = []
print("Dataset...", end=" ")
X, y = None, None
for S in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]:
    if (X is None):
        X = copy.deepcopy(DSX[S])
        y = copy.deepcopy(DSy[S])
    else:
        X = np.concatenate([X, DSX[S]], axis = 0)
        y = np.concatenate([y, DSy[S]], axis = 0)
print("loaded", end=" ")
Xtr, Xvl, ytr, yvl = train_test_split(X, y, test_size = 0.1, train_size = 0.9, random_state=42)
print("and splitted")
print("Training in progress...", end=" ")
logdir = get_run_logdir("test")
es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, verbose = 0, restore_best_weights = True)
tb = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)
start = time.time()
graph = model.fit(Xtr, ytr, epochs = 100, validation_data = (Xvl, yvl), callbacks = [es, tb], verbose = 0)
end = time.time()
print("and done")
modeltime += (end - start)
history = model.history.history['val_loss']
epochs.append(np.argmin(history) + 1)
scores.append(model.evaluate(Xts, yts, verbose = 0)[1])
print('Numero epoche:', np.around(np.mean(epochs), 2))
print('Media tempo di addestramento:', np.around(modeltime/5, 2), 's')
print('Accuracy:', np.around(np.mean(scores), 4)*100, '%')
print("Total model...", end=" ")
model.save('/home/fexed/models/autotest/total_ASCERTAINGRU')
print("saved")
ax.bar(1, np.around(np.mean(scores), 4)*100, yerr=0)

del X
del y
del DSX
del DSy

print("\nCONTINUAL TRAINING")
print("Model...", end=" ")
model = create_model()
print("created")

R = []
T = 4
b = [model.evaluate(Xts_1, yts_1, verbose = 0)[1], model.evaluate(Xts_2, yts_2, verbose = 0)[1], model.evaluate(Xts_3, yts_3, verbose = 0)[1], model.evaluate(Xts_4, yts_4, verbose = 0)[1]]
ACC, BWT, FWT = 0, 0, 0
modeltime = 0
epochs = []
scores = []

print("Training\tstarted")
for S in [("S0", "S1"), ("S2", "S3"), ("S4", "S5"), ("S6", "S7"), ("S8", "S9"), ("S10", "S11"), ("S12", "S13"), ("S14", "S15")]:
    print("Subjects\t" + S[0] + " " + S[1], end="\r")
    Xa = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/X" + S[0] + ".pkl", 'rb'), encoding='latin1')
    ya = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/y" + S[0] + ".pkl", 'rb'), encoding='latin1')
    Xb = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/X" + S[1] + ".pkl", 'rb'), encoding='latin1')
    yb = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/y" + S[1] + ".pkl", 'rb'), encoding='latin1')

    X = np.concatenate([Xa, Xb], axis = 0)
    y = np.concatenate([ya, yb], axis = 0)
    del Xa
    del Xb
    del ya
    del yb

    Xtr, Xvl, ytr, yvl = train_test_split(X, y, test_size = 0.1, train_size = 0.9, random_state=42)
    logdir = get_run_logdir(S[0] + S[1] + "_continual")
    es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, verbose = 0, restore_best_weights = True)
    tb = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)
    start = time.time()
    graph = model.fit(Xtr, ytr, epochs = 100, validation_data = (Xvl, yvl), callbacks = [es, tb], verbose = 0)
    end = time.time()
    modeltime += (end - start)
    history = model.history.history['val_loss']
    epochs.append(np.argmin(history) + 1)
    scores.append(model.evaluate(Xts, yts, verbose = 0)[1])

    R.append([model.evaluate(Xts_1, yts_1, verbose = 0)[1], model.evaluate(Xts_2, yts_2, verbose = 0)[1], model.evaluate(Xts_3, yts_3, verbose = 0)[1], model.evaluate(Xts_4, yts_4, verbose = 0)[1]])

print("Training\tended  ")
t = 0
for i in range(T):
        t += R[T-1][i]
ACC = t/T

t = 0
for i in range(T-1):
        t += (R[T-1][i] - R[i][i])
BWT = t/(T-1)

t = 0
for i in range(1, T):
        t += (R[i-1][i] - b[i])
FWT = t/(T-1)
print('Media numero epoche:', np.around(np.mean(epochs), 2))
print('Deviazione standard numero epoche:', np.around(np.std(epochs), 2))
print('Media tempo di addestramento:', np.around(modeltime/5, 2), 's')
print('Accuracy:', str(list(scores)), '%')
print('Media accuracy:', np.around(np.mean(scores), 4)*100, '%')
print('Deviazione standard accuracy:', np.around(np.std(scores), 4)*100, '%')
print('ACC\t', np.around(ACC, 4))
print('BWT\t', np.around(BWT, 4))
print('FWT\t', np.around(FWT, 4))
print("Continual model...", end=" ")
model.save('/home/fexed/ML/models/autotest/continual_ASCERTAINGRU')
print("saved")
ax.bar(2, np.around(np.mean(scores), 4)*100, yerr=np.around(np.std(scores), 4)*100)
del X
del y
del Xtr
del Xvl
del ytr
del yvl

print("\nCUMULATIVE TRAINING")
print("Model...", end=" ")
model = create_model()
print("created")

R = []
T = 4
b = [model.evaluate(Xts_1, yts_1, verbose = 0)[1], model.evaluate(Xts_2, yts_2, verbose = 0)[1], model.evaluate(Xts_3, yts_3, verbose = 0)[1], model.evaluate(Xts_4, yts_4, verbose = 0)[1]]
ACC, BWT, FWT = 0, 0, 0
modeltime = 0
epochs = []
scores = []
X, y = None, None

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
    graph = model.fit(Xtr, ytr, epochs = 100, validation_data = (Xvl, yvl), callbacks = [es, tb], verbose = 0)
    end = time.time()
    modeltime += (end - start)
    history = model.history.history['val_loss']
    epochs.append(np.argmin(history) + 1)
    scores.append(model.evaluate(Xts, yts, verbose = 0)[1])

    R.append([model.evaluate(Xts_1, yts_1, verbose = 0)[1], model.evaluate(Xts_2, yts_2, verbose = 0)[1], model.evaluate(Xts_3, yts_3, verbose = 0)[1], model.evaluate(Xts_4, yts_4, verbose = 0)[1]])

print("Training\tended  ")
t = 0
for i in range(T):
        t += R[T-1][i]
ACC = t/T

t = 0
for i in range(T-1):
        t += (R[T-1][i] - R[i][i])
BWT = t/(T-1)

t = 0
for i in range(1, T):
        t += (R[i-1][i] - b[i])
FWT = t/(T-1)
print('Media numero epoche:', np.around(np.mean(epochs), 2))
print('Deviazione standard numero epoche:', np.around(np.std(epochs), 2))
print('Media tempo di addestramento:', np.around(modeltime/5, 2), 's')
print('Accuracy:', str(list(scores)), '%')
print('Media accuracy:', np.around(np.mean(scores), 4)*100, '%')
print('Deviazione standard accuracy:', np.around(np.std(scores), 4)*100, '%')
print('ACC\t', np.around(ACC, 4))
print('BWT\t', np.around(BWT, 4))
print('FWT\t', np.around(FWT, 4))
print("Cumulative model...", end=" ")
model.save('/home/fexed/ML/models/autotest/cumulative_ASCERTAINGRU')
print("saved")
ax.bar(3, np.around(np.mean(scores), 4)*100, yerr=np.around(np.std(scores), 4)*100)
del X
del y
del Xtr
del Xvl
del ytr


print("\n25% REPLAY TRAINING")
print("Model...", end=" ")
model = create_model()
print("created")

R = []
T = 4
b = [model.evaluate(Xts_1, yts_1, verbose = 0)[1], model.evaluate(Xts_2, yts_2, verbose = 0)[1], model.evaluate(Xts_3, yts_3, verbose = 0)[1], model.evaluate(Xts_4, yts_4, verbose = 0)[1]]
ACC, BWT, FWT = 0, 0, 0
modeltime = 0
epochs = []
scores = []
X, y = None, None

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
    logdir = get_run_logdir(S[0] + S[1] + "_replay")
    es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, verbose = 0, restore_best_weights = True)
    tb = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)
    start = time.time()
    graph = model.fit(Xtr, ytr, epochs = 100, validation_data = (Xvl, yvl), callbacks = [es, tb], verbose = 0)
    end = time.time()
    modeltime += (end - start)
    history = model.history.history['val_loss']
    epochs.append(np.argmin(history) + 1)
    scores.append(model.evaluate(Xts, yts, verbose = 0)[1])

    R.append([model.evaluate(Xts_1, yts_1, verbose = 0)[1], model.evaluate(Xts_2, yts_2, verbose = 0)[1], model.evaluate(Xts_3, yts_3, verbose = 0)[1], model.evaluate(Xts_4, yts_4, verbose = 0)[1]])

    X, _, y, _ = train_test_split(Xtr, ytr, test_size = 0.75, train_size = 0.25, random_state=42)

print("Training\tended  ")
t = 0
for i in range(T):
        t += R[T-1][i]
ACC = t/T

t = 0
for i in range(T-1):
        t += (R[T-1][i] - R[i][i])
BWT = t/(T-1)

t = 0
for i in range(1, T):
        t += (R[i-1][i] - b[i])
FWT = t/(T-1)
print('Media numero epoche:', np.around(np.mean(epochs), 2))
print('Deviazione standard numero epoche:', np.around(np.std(epochs), 2))
print('Media tempo di addestramento:', np.around(modeltime/5, 2), 's')
print('Accuracy:', str(list(scores)), '%')
print('Media accuracy:', np.around(np.mean(scores), 4)*100, '%')
print('Deviazione standard accuracy:', np.around(np.std(scores), 4)*100, '%')
print('ACC\t', np.around(ACC, 4))
print('BWT\t', np.around(BWT, 4))
print('FWT\t', np.around(FWT, 4))
print("Replay model...", end=" ")
model.save('/home/fexed/ML/models/autotest/replay_ASCERTAINGRU')
print("saved")
ax.bar(4, np.around(np.mean(scores), 4)*100, yerr=np.around(np.std(scores), 4)*100)

print("\nEPISODIC TRAINING (m = 70)")
print("Model...", end=" ")
model = create_model()
print("created")

R = []
T = 4
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
    logdir = get_run_logdir(S[0] + S[1] + "_episodic")
    es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, verbose = 1, restore_best_weights = True)
    tb = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)
    start = time.time()
    graph = model.fit(Xtr, ytr, epochs = 100, validation_data = (Xvl, yvl), callbacks = [es, tb], verbose = 0)
    end = time.time()
    modeltime += (end - start)
    history = model.history.history['val_loss']
    epochs.append(np.argmin(history) + 1)
    scores.append(model.evaluate(Xts, yts, verbose = 0)[1])

    R.append([model.evaluate(Xts_1, yts_1, verbose = 0)[1], model.evaluate(Xts_2, yts_2, verbose = 0)[1], model.evaluate(Xts_3, yts_3, verbose = 0)[1], model.evaluate(Xts_4, yts_4, verbose = 0)[1]])

    idx_1 = [i for i, x in enumerate(ytr) if x[0] == 1]
    for i in idx_1:
        X_1.append(Xtr[i])
        y_1.append([1., 0., 0., 0.])
    random.shuffle(X_1)
    X_1 = X_1[0:m]
    y_1 = y_1[0:m]
    idx_2 = [i for i, x in enumerate(ytr) if x[1] == 1]
    for i in idx_2:
        X_2.append(Xtr[i])
        y_2.append([0., 1., 0., 0.])
    random.shuffle(X_2)
    X_2 = X_2[0:m]
    y_2 = y_2[0:m]
    idx_3 = [i for i, x in enumerate(ytr) if x[2] == 1]
    for i in idx_3:
        X_3.append(Xtr[i])
        y_3.append([0., 0., 1., 0.])
    random.shuffle(X_3)
    X_3 = X_3[0:m]
    y_3 = y_3[0:m]
    idx_4 = [i for i, x in enumerate(ytr) if x[3] == 1]
    for i in idx_4:
        X_4.append(Xtr[i])
        y_4.append([0., 0., 0., 1.])
    random.shuffle(X_4)
    X_4 = X_4[0:m]
    y_4 = y_4[0:m]

    X = np.concatenate([X_1, X_2, X_3, X_4], axis = 0)
    y = np.concatenate([y_1, y_2, y_3, y_4], axis = 0)

print("Training\tended  ")
t = 0
for i in range(T):
        t += R[T-1][i]
ACC = t/T

t = 0
for i in range(T-1):
        t += (R[T-1][i] - R[i][i])
BWT = t/(T-1)

t = 0
for i in range(1, T):
        t += (R[i-1][i] - b[i])
FWT = t/(T-1)
print('Media numero epoche:', np.around(np.mean(epochs), 2))
print('Deviazione standard numero epoche:', np.around(np.std(epochs), 2))
print('Media tempo di addestramento:', np.around(modeltime/5, 2), 's')
print('Accuracy:', str(list(scores)), '%')
print('Media accuracy:', np.around(np.mean(scores), 4)*100, '%')
print('Deviazione standard accuracy:', np.around(np.std(scores), 4)*100, '%')
print('ACC\t', np.around(ACC, 4))
print('BWT\t', np.around(BWT, 4))
print('FWT\t', np.around(FWT, 4))
print("Episodic model...", end=" ")
model.save('/home/fexed/ML/models/autotest/episodic_ASCERTAINGRU')
print("saved")
ax.bar(5, np.around(np.mean(scores), 4)*100, yerr=np.around(np.std(scores), 4)*100)

print("Plot...", end=" ")
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(['Offline', 'Continual', 'Cumulative', 'Replay', '70 Examples'])
ax.set_ylabel("Accuracy")
ax.set_ylim([0, 100])
plt.savefig("plots/autotest_ASCERTAIN_GRU.png")
print("done")
