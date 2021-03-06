import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import copy


# Log directory
def get_run_logdir(label=""):
    import time
    run_id = time.strftime("continual_wesad_%Y%m%d_%H%M%S")
    return os.path.join("/home/fexed/ML/tensorboard_logs", run_id + "_" + label)


print(os.getpid())
input("Press Enter to continue...")
print("Opening test set")
Xts = pickle.load(open("/home/fexed/ML/datasets/WESAD/splitted/XS17_disarli.pkl", 'rb'), encoding='latin1')
yts = pickle.load(open("/home/fexed/ML/datasets/WESAD/splitted/yS17_disarli.pkl", 'rb'), encoding='latin1')

print("Creating model")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.GRU(30, input_shape=(100, 14)))
#model.add(tf.keras.layers.LSTM(30, return_sequences = True, input_shape=(100, 14)))
#model.add(tf.keras.layers.LSTM(30))
model.add(tf.keras.layers.Dense(4, activation = 'softmax',
                  kernel_regularizer = tf.keras.regularizers.l1_l2(l1 = 1e-6, l2 = 1e-6),
                  bias_regularizer = tf.keras.regularizers.l1_l2(l1 = 1e-4, l2 = 1e-4),
                  activity_regularizer = tf.keras.regularizers.l1_l2(l1 = 1e-4, l2 = 1e-4)))
opt = tf.keras.optimizers.Adam(learning_rate = 0.005, beta_1 = 0.85, beta_2 = 0.999)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

modeltime = 0
epochs = []
scores = []
X, y = None, None

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
    logdir = get_run_logdir(S[0] + S[1] + "_sum")
    es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, verbose = 1, restore_best_weights = True)
    tb = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)
    start = time.time()
    graph = model.fit(Xtr, ytr, epochs = 100, validation_data = (Xvl, yvl), callbacks = [es, tb])
    end = time.time()
    modeltime += (end - start)
    history = model.history.history['val_loss']
    epochs.append(np.argmin(history) + 1)
    scores.append(model.evaluate(Xts, yts)[1])

print('Media numero epoche:', np.around(np.mean(epochs), 2))
print('Deviazione standard numero epoche:', np.around(np.std(epochs), 2))
print('Media tempo di addestramento:', np.around(modeltime/5, 2), 's')
print('Accuracy:', str(list(scores)), '%')
print('Media accuracy:', np.around(np.mean(scores), 4)*100, '%')
print('Deviazione standard accuracy:', np.around(np.std(scores), 4)*100, '%')
plt.plot(scores)
plt.savefig("accuracy_sum.png")
