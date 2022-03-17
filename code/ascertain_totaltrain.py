import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
import copy


# Log directory
def get_run_logdir(label=""):
    import time
    run_id = time.strftime("total_ASCERTAIN_%Y%m%d_%H%M%S")
    return os.path.join("/home/fexed/ML/tensorboard_logs", run_id + "_" + label)


print(os.getpid())
input("Press Enter to continue...")

print("Creating model")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.GRU(16, return_sequences = True, input_shape=(160, 17)))
model.add(tf.keras.layers.GRU(16))
model.add(tf.keras.layers.Dense(4, activation = 'softmax',
                  kernel_regularizer = tf.keras.regularizers.l1_l2(l1 = 1e-5, l2 = 1e-5),
                  bias_regularizer = tf.keras.regularizers.l1_l2(l1 = 1e-4, l2 = 1e-4),
                  activity_regularizer = tf.keras.regularizers.l1_l2(l1 = 1e-6, l2 = 1e-6)))
opt = tf.keras.optimizers.Adam(learning_rate = 0.01, beta_1 = 0.95, beta_2 = 0.999)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

modeltime = 0
epochs = []
scores = []
Xtr, ytr = None, None
for S in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]:
    Xs = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/X" + S + ".pkl", 'rb'), encoding='latin1')
    ys = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/y" + S + ".pkl", 'rb'), encoding='latin1')

    if (Xtr is None):
        Xtr = copy.deepcopy(Xs)
        ytr = copy.deepcopy(ys)
    else:
        Xtr = np.concatenate([Xtr, Xs], axis = 0)
        ytr = np.concatenate([ytr, ys], axis = 0)
    del Xs
    del ys
    print("Loaded " + S)

Xts = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/Xts.pkl", 'rb'), encoding='latin1')
yts = pickle.load(open("/home/fexed/ML/datasets/ASCERTAIN/splitted/yts.pkl", 'rb'), encoding='latin1')
print("Dataset loaded")

Xtr, Xvl, ytr, yvl = train_test_split(Xtr, ytr, test_size = 0.25, train_size = 0.75, random_state=42)
print("Dataset splitted")
print(str(Xtr.shape) + " " + str(Xvl.shape))
logdir = get_run_logdir("all")
es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, verbose = 1, restore_best_weights = True)
tb = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)
start = time.time()
graph = model.fit(Xtr, ytr, epochs = 100, validation_data = (Xvl, yvl), callbacks = [es, tb], batch_size=256, use_multiprocessing=True, workers=8)
end = time.time()
modeltime += (end - start)
history = model.history.history['val_loss']
epochs.append(np.argmin(history) + 1)
scores.append(model.evaluate(Xts, yts)[1])

print('Epoche:', np.around(np.mean(epochs), 2))
print('Tempo di addestramento:', np.around(modeltime, 2), 's')
print('Accuracy:', np.around(np.mean(scores), 4)*100, '%')
