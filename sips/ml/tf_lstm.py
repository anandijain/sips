import os
import datetime
import random

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sips.macros import macros as m
from sips.macros import bov as bm
from sips.lines.bov import bov
from sips.h import fileio as fio
from sips.h import helpers as h
from sips.h import serialize as s
from sips.h import hot
from sips.h import viz
from sips.h import tf_loaders as tfls


class TfLSTM(tf.keras.Model):
    """
    subclassing model type
    """

    def __init__(self, in_dim):
        super(TfLSTM, self).__init__()
        # self.e1 = tf.keras.layers.Embedding(input_dim=in_shape, output_dim=64)
        self.l1 = tf.keras.layers.LSTM(
            100, activation="relu", input_shape=(None, in_dim), return_sequences=True)
        self.l2 = tf.keras.layers.LSTM(128, activation="relu")
        self.l3 = tf.keras.layers.Dense(19, activation="softmax")

    def call(self, x):
        print(x.shape)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


def make_model():
    # sequential model

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.LSTM(100, input_shape=(
                None, 202), return_sequences=True, activation='relu'),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(19, activation="softmax"),
        ]
    )
    return model


def make_model_functional(in_shape_tup):
    inputs = tf.keras.Input(shape=in_shape_tup, batch_size=1)
    x = tf.keras.layers.LSTM(64, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def get_loss_metrics():
    train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean("test_loss", dtype=tf.float32)
    return train_loss, test_loss


def get_acc_metrics():
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        "train_accuracy")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy("test_accuracy")
    return train_accuracy, test_accuracy


def get_classification_metrics():
    train_loss, test_loss = get_loss_metrics()
    train_accuracy, test_accuracy = get_acc_metrics()
    return train_loss, train_accuracy, test_loss, test_accuracy


def train_step_classify(model, optimizer, loss_object, x_train, y_train, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
        maxed_pred = tf.argmax(predictions, 1).numpy()[0]
        maxed_true = tf.argmax(y_train).numpy()
        # assumes batch size 1
        correct = tf.equal(maxed_pred, maxed_true).numpy()
        strs = bm.TRANSITION_CLASS_STRINGS
        # print(f"preds: {maxed_pred}")
        # print(f"actuals: {maxed_true}")

        # print(f"preds_str: {strs[maxed_pred]}")
        # print(f"actual_str: {strs[maxed_true]}")
        print(loss.numpy())
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    tl = train_loss(loss)
    ta = train_accuracy(y_train, predictions)
    return tl, ta, correct


def train_step_regress(model, optimizer, loss_object, x_train, y_train, train_loss):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
        print(
            f'preds: {predictions}, actuals: {y_train}, loss: {loss.numpy()}')
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    tl = train_loss(loss)
    return tl


def test_step(model, loss_object, x_test, y_test, test_loss, test_accuracy=None):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)

    te_loss = test_loss(loss)
    if test_accuracy:
        tea = test_accuracy(y_test, predictions)
        return te_loss, tea
    else:
        return te_loss


def init_summary_writers():
    """

    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = "logs/gradient_tape/" + current_time + "/train"
    test_log_dir = "logs/gradient_tape/" + current_time + "/test"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    return train_summary_writer, test_summary_writer


def model_core():
    model = make_model()
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    return model, loss_object, optimizer


def train_directional_predictor(datasets, test_datasets):
    # EPOCHS = 10
    BATCH_SIZE = 1
    BUFFER_SIZE = 100
    datasets, test_datasets = tfls.get_directional_datasets()
    model, loss_fxn, optim = model_core()
    train_summary_writer, test_summary_writer = init_summary_writers()
    train_loss, train_accuracy, test_loss, test_accuracy = get_classification_metrics()
    for epoch, dataset in enumerate(datasets):
        if not dataset:
            continue
        for (xtr, ytr) in dataset:
            tl, ta, correct = train_step_classify(
                model, optim, loss_fxn, xtr, ytr, train_loss, train_accuracy)

            if correct.any():
                print("guessed correctly")
            else:
                print("guessed wrong")

        with train_summary_writer.as_default():

            tf.summary.scalar("loss", tl.numpy(), step=epoch)
            tf.summary.scalar("accuracy", ta.numpy(), step=epoch)

        test_dataset = random.choice(test_datasets)

        if not test_dataset:
            continue
        for (xte, yte) in test_dataset:
            tel, tea = test_step(model, loss_fxn, xte, yte,
                                 test_loss, test_accuracy)

        with test_summary_writer.as_default():
            tf.summary.scalar("loss", tel.numpy(), step=epoch)
            tf.summary.scalar("accuracy", tea.numpy(), step=epoch)

        template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
        print(
            template.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result() * 100,
                test_loss.result(),
                test_accuracy.result() * 100,
            )
        )

        # Reset metrics every epoch
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

    tf.saved_model.save(model, "./logs/models/1/")


def get_example(datasets):
    for epoch, dataset in enumerate(datasets):

        if not dataset:
            continue
        # data = dataset.batch(1)

        for (x_train, y_train) in dataset:
            print(f'x_train: {x_train}')
            print(f'x_train.shape: {x_train.shape}')
            print(f'y_train: {y_train}')
            print(f'y_train.shape: {y_train.shape}')
            break
        break
    return x_train, y_train


def main():
    '''

    '''
    folder = m.PROJ_DIR + "ml/lines/"
    all_datasets = tfls.get_pred_datasets(folder, label_cols=[
                                          'a_ml', 'h_ml'], batch_size=1, buffer_size=10, history_size=100, pred_size=1, step_size=1, norm=True)
    datasets, test_datasets = h.train_test_split_list(all_datasets)
    x, y = get_example(datasets)

    loss_fxn = tf.losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.RMSprop(clipvalue=1.0)
    model = make_model_functional(x.shape[-2:])
    train_loss, test_loss = get_loss_metrics()

    train_summary_writer, test_summary_writer = init_summary_writers()

    for epoch, dataset in enumerate(datasets):
        if not dataset:
            continue

        for (x_train, y_train) in dataset:
            tl = train_step_regress(
                model, optimizer, loss_fxn, x_train, y_train, train_loss)

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", tl.numpy(), step=epoch)

        test_dataset = random.choice(test_datasets)
        
        if not test_dataset:
            continue

        for (xte, yte) in test_dataset:
            tel = test_step(model, loss_fxn, xte, yte,
                            test_loss)

        with test_summary_writer.as_default():
            tf.summary.scalar("loss", tel.numpy(), step=epoch)

        if epoch % 2000:
            template = "Epoch {}, Loss: {}, Test Loss: {}"
            print(
                template.format(
                    epoch + 1,
                    train_loss.result(),
                    test_loss.result(),
                )
            )

        # Reset metrics every epoch
        train_loss.reset_states()
        test_loss.reset_states()

    tf.saved_model.save(model, "./logs/models/ml_pred/")


if __name__ == "__main__":
    # main()
    main()
