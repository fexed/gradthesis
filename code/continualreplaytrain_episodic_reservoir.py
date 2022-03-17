import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
import random
import matplotlib.pyplot as plt
import DeepRC


# Log directory
def get_run_logdir(label=""):
    import time
    run_id = time.strftime("continual_episodic_wesad_%Y%m%d_%H%M%S")
    return os.path.join("/home/fexed/ML/tensorboard_logs", run_id + "_" + label)


print("Opening test set")
Xts = pickle.load(open("/home/fexed/ML/datasets/WESAD/splitted/XS17_disarli.pkl", 'rb'), encoding='latin1')
yts = pickle.load(open("/home/fexed/ML/datasets/WESAD/splitted/yS17_disarli.pkl", 'rb'), encoding='latin1')
print("Creating test set per task")
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


print("Creating model")
model = DeepRC.SimpleDeepESNClassifier(num_classes = 4, units = 37, layers = 1)
opt = tf.keras.optimizers.Adam(learning_rate = 0.005, beta_1 = 0.85, beta_2 = 0.999)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

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

for S in [("S2", "S3"), ("S4", "S5"), ("S6", "S7"), ("S8", "S9"), ("S10", "S11"), ("S13", "S14"), ("S15", "S16")]:
    print("Subjects " + S[0] + " and " + S[1])
    Xa = pickle.load(open("/home/fexed/ML/datasets/WESAD/splitted/X" + S[0] + "_disarli.pkl", 'rb'), encoding='latin1')
    ya = pickle.load(open("/home/fexed/ML/datasets/WESAD/splitted/y" + S[0] + "_disarli.pkl", 'rb'), encoding='latin1')
    Xb = pickle.load(open("/home/fexed/ML/datasets/WESAD/splitted/X" + S[1] + "_disarli.pkl", 'rb'), encoding='latin1')
    yb = pickle.load(open("/home/fexed/ML/datasets/WESAD/splitted/y" + S[1] + "_disarli.pkl", 'rb'), encoding='latin1')

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
    logdir = get_run_logdir(S[0] + S[1])
    es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, restore_best_weights = True)
    #tb = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)
    start = time.time()
    graph = model.fit(Xtr, ytr, epochs = 100, validation_data = (Xvl, yvl), callbacks = [es])
    end = time.time()
    modeltime += (end - start)
    history = model.history.history['val_loss']
    epochs.append(np.argmin(history) + 1)
    scores.append(model.evaluate(Xts, yts)[1])

    R.append([model.evaluate(Xts_1, yts_1, verbose = 0)[1], model.evaluate(Xts_2, yts_2, verbose = 0)[1], model.evaluate(Xts_3, yts_3, verbose = 0)[1], model.evaluate(Xts_4, yts_4, verbose = 0)[1]])

    #X, _, y, _ = train_test_split(Xtr, ytr, test_size = 0.75, train_size = 0.25, random_state=42)  # replay random di elementi precedenti
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
#plt.plot(scores)
#plt.savefig("accuracy_continual_episodic.png")
#model.save("/home/fexed/ML/models/replaymodel_episodic")
