import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import ParameterSampler, train_test_split
import time
import copy


# Log directory
def get_run_logdir(label=""):
    import time
    run_id = time.strftime("total_wesad_%Y%m%d_%H%M%S")
    return os.path.join("/home/fexed/ML/tensorboard_logs", run_id + "_" + label)


def createModelGRU(num_classes = 3, units = 18, layers = 1,
                   learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999,
                   kernel_regularizer = 1e-5,
                   bias_regularizer = 1e-5,
                   activity_regularizer = 1e-5):
    model = tf.keras.models.Sequential()
    for i in range(0, layers - 1):
        model.add(tf.keras.layers.GRU(units, return_sequences = True))
    model.add(tf.keras.layers.GRU(units))
    model.add(tf.keras.layers.Dense(num_classes, activation = 'softmax',
                  kernel_regularizer = tf.keras.regularizers.l1_l2(l1 = kernel_regularizer, l2 = kernel_regularizer),
                  bias_regularizer = tf.keras.regularizers.l1_l2(l1 = bias_regularizer, l2 = bias_regularizer),
                  activity_regularizer = tf.keras.regularizers.l1_l2(l1 = activity_regularizer, l2 = activity_regularizer)))
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate, beta_1 = beta_1, beta_2 = beta_2)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    return model

Xts = pickle.load(open("/home/fexed/ML/datasets/WESAD/splitted/XS17_disarli.pkl", 'rb'), encoding='latin1')
yts = pickle.load(open("/home/fexed/ML/datasets/WESAD/splitted/yS17_disarli.pkl", 'rb'), encoding='latin1')
modeltime = 0
epochs = []
scores = []
X, y = [], []
for S in ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16"]:
    Xs = pickle.load(open("/home/fexed/ML/datasets/WESAD/splitted/X" + S + "_disarli.pkl", 'rb'), encoding='latin1')
    ys = pickle.load(open("/home/fexed/ML/datasets/WESAD/splitted/y" + S + "_disarli.pkl", 'rb'), encoding='latin1')

    if (X == []):
        X = copy.deepcopy(Xs)
        y = copy.deepcopy(ys)
    else:
        X = np.concatenate([X, Xs], axis = 0)
        y = np.concatenate([y, ys], axis = 0)
    del Xs
    del ys
    print("Loaded " + S)

print(X.shape)
print(y.shape)
print("Both datasets loaded")

Xtr, Xvl, ytr, yvl = train_test_split(X, y, test_size = 0.25, train_size = 0.75, random_state=42)
print("Dataset splitted")
print(str(Xtr.shape) + " " + str(Xvl.shape))
modelGRU = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn = createModelGRU)
kernel_regularizer = [1e-4, 1e-5, 1e-6]
bias_regularizer = [1e-4, 1e-5, 1e-6]
activity_regularizer = [1e-4, 1e-5, 1e-6]
layers = [1, 2, 3]
units = list(range(15, 36))
param_distr = {'kernel_regularizer': kernel_regularizer,
               'bias_regularizer': bias_regularizer,
               'activity_regularizer': activity_regularizer,
               'layers': layers,
               'units': units}
best_score = -1
for g in ParameterSampler(param_distr, n_iter = 10000):
    modelGRU.set_params(**g, num_classes = 4, learning_rate = 0.005, beta_1 = 0.99, beta_2 = 0.99)
    modelGRU.fit(X, y, verbose = 0)
    #Save if best
    score = modelGRU.score(Xts, yts, verbose = 0)
    if score > best_score:
        best_score = score
        best_params = g
        print("Current best: %f using %s" % (best_score, best_params))
print("Best: %f using %s" % (best_score, best_params))
