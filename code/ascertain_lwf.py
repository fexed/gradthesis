import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import lwf_train


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
b = [model.evaluate(Xts_1, yts_1, verbose = 0)[1], model.evaluate(Xts_2, yts_2, verbose = 0)[1], model.evaluate(Xts_3, yts_3, verbose = 0)[1], model.evaluate(Xts_4, yts_4, verbose = 0)[1]]
ACC, BWT, FWT = 0, 0, 0
modeltime = 0
epochs = []
scores = []

for S in [("S0", "S1"), ("S2", "S3"), ("S4", "S5"), ("S6", "S7"), ("S8", "S9"), ("S10", "S11"), ("S12", "S13"), ("S14", "S15")]:
    print("Subjects " + S[0] + " and " + S[1])
    Xa = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/X" + S[0] + ".pkl", 'rb'), encoding='latin1')
    ya = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/y" + S[0] + ".pkl", 'rb'), encoding='latin1')
    Xb = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/X" + S[1] + ".pkl", 'rb'), encoding='latin1')
    yb = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/y" + S[1] + ".pkl", 'rb'), encoding='latin1')

    X = np.concatenate([Xa, Xb], axis = 0)
    y = np.concatenate([ya, yb], axis = 0)
    print(str(Xa.shape) + " + " + str(Xb.shape) + " = " + str(X.shape))
    print(str(ya.shape) + " + " + str(yb.shape) + " = " + str(y.shape))
    print("Both datasets loaded")
    del Xa
    del Xb
    del ya
    del yb

    Xtr, Xvl, ytr, yvl = train_test_split(X, y, test_size = 0.1, train_size = 0.9, random_state=42)
    print("Dataset splitted")
    print(str(Xtr.shape) + " " + str(Xvl.shape))
    logdir = get_run_logdir(S[0] + S[1])
    es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, verbose = 1, restore_best_weights = True)
    tb = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)
    start = time.time()
    nepochs = lwf_train.train_lwf(model, Xtr, ytr, Xvl, yvl, epochs = 100, callbacks = [es, tb])
    end = time.time()
    modeltime += (end - start)
    #history = model.history.history['val_loss']
    epochs.append(nepochs + 1)
    scores.append(model.evaluate(Xts, yts)[1])

    R.append([model.evaluate(Xts_1, yts_1, verbose = 0)[1], model.evaluate(Xts_2, yts_2, verbose = 0)[1], model.evaluate(Xts_3, yts_3, verbose = 0)[1], model.evaluate(Xts_4, yts_4, verbose = 0)[1]])

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
print('Media tempo di addestramento:', np.around(modeltime/8, 2), 's')
print('Accuracy:', str(list(scores)), '%')
print('Media accuracy:', np.around(np.mean(scores), 4)*100, '%')
print('Deviazione standard accuracy:', np.around(np.std(scores), 4)*100, '%')
print('ACC\t', np.around(ACC, 4))
print('BWT\t', np.around(BWT, 4))
print('FWT\t', np.around(FWT, 4))
#plt.plot(scores)
#plt.savefig("accuracy_continual.png")
