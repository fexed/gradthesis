# https://seanmoriarity.com/2020/10/18/continual-learning-with-ewc/

import tensorflow as tf
import numpy as np


def precision_matrices(model, Xtr, ytr, n_batches=1, batch_size=32):
    prec_matr = {n: tf.zeros_like(p.value()) for n, p in enumerate(model.trainable_variables)}

    for i, x in enumerate(Xtr):
        y = ytr[i]
        with tf.GradientTape() as tape:
            preds = model(x.reshape(-1, 160, 17))
            ll = tf.nn.log_softmax(preds)
        ll_grads = tape.gradient(ll, model.trainable_variables)

        for i, g in enumerate(ll_grads):
            prec_matr[i] += tf.math.reduce_mean(g ** 2, axis = 0) / n_batches

    return prec_matr


def elastic_penalty(F, theta, theta_A, alpha=25):
    penalty = 0
    for i, theta_i in enumerate(theta):
        tmp = tf.math.reduce_sum(F[i] * (theta_i - theta_A[i]) ** 2)
        penalty += tmp

    return 0.5 * alpha * penalty


def ewc_loss(y, preds, model, F, theta_A):
    loss_b = tf.keras.losses.get(model.loss)(y, preds)
    penalty = elastic_penalty(F, model.trainable_variables, theta_A)
    return loss_b + penalty


def train_ewc(model, Xtr, ytr, Xvl, yvl, epochs=10, callbacks=None, F=None, theta_A=None):
    if (F is None or theta_A is None):  # first training data in
        history = model.fit(Xtr, ytr, validation_data = (Xvl, yvl), epochs=epochs, callbacks=callbacks)
        theta_A = {n: p.value() for n, p in enumerate(model.trainable_variables.copy())}
        F = precision_matrices(model, Xtr, ytr, n_batches=1000)
        return F, theta_A, np.argmin(history)
    else:  # custom fit for continual data
        #for callback in callbacks:
            #callback.on_train_begin()
        accuracy = tf.keras.metrics.CategoricalAccuracy('accuracy')
        loss = tf.keras.metrics.CategoricalCrossentropy('loss')
        c = 0
        minloss = 1e10

        for epoch in range(epochs):
            #for callback in callbacks:
                #callback.on_epoch_begin(epoch)
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
                    #for callback in callbacks:
                        #callback.on_train_batch_begin((i+1)%32)
                    xbatch = np.array(xbatch)
                    ybatch = np.array(ybatch)
                    with tf.GradientTape() as tape:
                        preds = model(xbatch)
                        total_loss = ewc_loss(ybatch, preds, model, F, theta_A)
                    grads = tape.gradient(total_loss, model.trainable_variables)
                    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    accuracy.update_state(ybatch, preds)
                    loss.update_state(ybatch, preds)
                    lss, acc = loss.result().numpy(), accuracy.result().numpy()
                    print("- loss: " + str(lss), end=" ")
                    print("- accuracy: " + str(acc), end="\r")
                    xbatch, ybatch = [], []
                    #for callback in callbacks:
                        #callback.on_train_batch_ends()
            print("")
            res = model.evaluate(Xvl, yvl)
            lss = res[0]
            # print(lss)
            if (minloss >= lss):  #early_stopping
                minloss = lss
                c = 0
            else:
                c += 1
                if c == 10:  # patience = 10
                    break;
            #for callback in callbacks:
                #callback.on_epoch_end(epoch)
        theta_A = {n: p.value() for n, p in enumerate(model.trainable_variables.copy())}
        F = precision_matrices(model, Xtr, ytr, n_batches=1000)
        return F, theta_A, epoch  # return data for next step
