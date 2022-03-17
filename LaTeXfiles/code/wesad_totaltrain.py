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
    run_id = time.strftime("total_wesad_%Y%m%d_%H%M%S")
    return os.path.join("tensorboard_logs", run_id + "_" + label)


print(os.getpid())
input("Press Enter to continue...")

print("Creating model")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.GRU(18, return_sequences=True, input_shape=(100, 14)))
model.add(tf.keras.layers.GRU(18))
model.add(tf.keras.layers.Dense(4, activation = 'softmax',
                  kernel_regularizer = tf.keras.regularizers.l1_l2(l1 = 1e-04, l2 = 1e-04),
                  bias_regularizer = tf.keras.regularizers.l1_l2(l1 = 1e-04, l2 = 1e-04),
                  activity_regularizer = tf.keras.regularizers.l1_l2(l1 = 1e-04, l2 = 1e-04)))
opt = tf.keras.optimizers.Adam(learning_rate = 0.005, beta_1 = 0.99, beta_2 = 0.99)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

modeltime = 0
epochs = []
scores = []
Xtr, ytr = None, None
for S in ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"]:
    Xs = pickle.load(open("datasets/WESAD/splitted/X" + S + ".pkl", 'rb'), encoding='latin1')
    ys = pickle.load(open("datasets/WESAD/splitted/y" + S + ".pkl", 'rb'), encoding='latin1')

    if (Xtr is None):
        Xtr = copy.deepcopy(Xs)
        ytr = copy.deepcopy(ys)
    else:
        Xtr = np.concatenate([Xtr, Xs], axis = 0)
        ytr = np.concatenate([ytr, ys], axis = 0)
    del Xs
    del ys
    print("Loaded " + S)


Xts = pickle.load(open("datasets/WESAD/splitted/Xts.pkl", 'rb'), encoding='latin1')
yts = pickle.load(open("datasets/WESAD/splitted/yts.pkl", 'rb'), encoding='latin1')
print(str(Xtr.shape) + " " + str(Xts.shape))
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

print('Media numero epoche:', np.around(np.mean(epochs), 2))
print('Media tempo di addestramento:', np.around(modeltime, 2), 's')
print('Media accuracy:', np.around(np.mean(scores), 4)*100, '%')
print('Deviazione standard accuracy:', np.around(np.std(scores), 4)*100, '%')
