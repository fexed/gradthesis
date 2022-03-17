import tensorflow as tf
import copy
import numpy as np


T = 2


def lwf_lossnew(ybatch, preds, model):
    lnew = 0
    for i, y in enumerate(ybatch):
        lnew += y * tf.math.log(preds[i])
    return -lnew


def lwf_lossold(ybatch, preds, model):
    lold = 0
    deny, denp = 0, 0
    for y1, y2 in zip(ybatch, preds):
        deny += (y1) ** (1/T)
        denp += (y2) ** (1/T)
    for i, y in enumerate(ybatch):
        lold += ((y ** (1/T))/deny) * ((preds[i] ** (1/T))/denp)
    return -lold


def train_lwf(model, Xtr, ytr, Xvl, yvl, epochs=10, callbacks=None):
    accuracy = tf.keras.metrics.CategoricalAccuracy('accuracy')
    loss = tf.keras.metrics.CategoricalCrossentropy('loss')
    prevmodel = tf.keras.models.clone_model(model)  # old model
    c = 0
    minloss = 1e10

    for epoch in range(epochs):
        print("Epoch " + str(epoch+1) + "/" + str(epochs))
        accuracy.reset_states()
        loss.reset_states()
        xbatch, ybatch = [], []
        max = str(len(Xtr)//32)
        for i, x in enumerate(Xtr):
            xbatch.append(x)
            ybatch.append(ytr[i])
            if ((i+1) % 32) == 0:
                print(str((i+1)//32) + "/" + max, end=" ")
                xbatch = np.array(xbatch)
                ybatch = np.array(ybatch)
                with tf.GradientTape() as tape:
                    preds = model(xbatch)
                    oldpreds = prevmodel(xbatch)
                    total_loss = lwf_lossnew(ybatch, preds, model) + lwf_lossold(ybatch, oldpreds, prevmodel) # + tf.keras.losses.get(model.loss)(ybatch, preds)
                grads = tape.gradient(total_loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                accuracy.update_state(ybatch, preds)
                loss.update_state(ybatch, preds)
                lss, acc = loss.result().numpy(), accuracy.result().numpy()
                print("- loss: " + str(lss), end=" ")
                print("- accuracy: " + str(acc), end="\r")
                xbatch, ybatch = [], []
        print("")
        res = model.evaluate(Xvl, yvl)
        lss = res[0]
        #print(lss)
        if (minloss > lss):  #early_stopping
            minloss = lss
            c = 0
        else:
            c += 1
            if c == 10:  # patience = 10
                break;
    return epoch
