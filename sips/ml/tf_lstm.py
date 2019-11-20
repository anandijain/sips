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
from sips.h import helpers as h
from sips.h import serialize as s
from sips.h import viz

COLS = [
    # "num_markets",
    # "live",
    'a_team',
    'h_team',
    "quarter",
    "secs",
    "a_pts",
    "h_pts",
    "status",
    "a_ps",
    "h_ps",
    "a_hcap",
    "h_hcap",
    "a_ml",
    "h_ml",
    # "game_start_time",
]


class TfLSTM(tf.keras.Model):
    """
    subclassing model type
    """

    def __init__(self, in_shape):
        super(TfLSTM, self).__init__()
        self.l1 = tf.keras.layers.LSTM(
            100, input_shape=in_shape, activation="relu")
        self.l2 = tf.keras.layers.LSTM(128, activation="relu")
        self.l3 = tf.keras.layers.Dense(19, activation="softmax")

    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


def make_model():
    # sequential model

    model = tf.keras.models.Sequential(
        [
            # tf.keras.layers.LSTM(100, input_shape = (15, 210), return_sequences=True, activation='relu'),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(19, activation="softmax"),
        ]
    )
    return model


def get_tf_dataset(fn, verbose=False):
    data = bov.prep_game_dataset(fn)
    if not data:
        return
    X, y = data
    if verbose:
        print(f"X: {X}, X[0].shape: {X[0].shape}")
        print(f"y: {y}")
    # tf_X = tf.convert_to_tensor(X)
    # tf_y = tf.convert_to_tensor(y)
    X = tf.keras.utils.normalize(X)

    dataset = tf.data.Dataset.from_tensor_slices((np.array(X), np.array(y)))

    return dataset


train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy("train_accuracy")
test_loss = tf.keras.metrics.Mean("test_loss", dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy("test_accuracy")


def train_step(model, optimizer, loss_object, x_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
        # predictions = tf.reshape(predictions, [-1])
        maxed_pred = tf.argmax(predictions, 1).numpy()[0]
        maxed_true = tf.argmax(y_train).numpy()
        # assumes batch size 1
        correct = tf.equal(maxed_pred, maxed_true).numpy()
        strs = bm.TRANSITION_CLASS_STRINGS
        print(f"preds: {maxed_pred}")
        print(f"actuals: {maxed_true}")

        print(f"preds_str: {strs[maxed_pred]}")
        print(f"actual_str: {strs[maxed_true]}")
        print(loss.numpy())
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    tl = train_loss(loss)
    ta = train_accuracy(y_train, predictions)
    return tl, ta, correct


def test_step(model, loss_object, x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)

    te = test_loss(loss)
    tea = test_accuracy(y_test, predictions)
    return te, tea


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


def get_directional_datasets():
    folder = m.PROJ_DIR + "ml/lines/"
    fns = h.get_fns(folder)
    train_fns, test_fns = h.train_test_split_dir(fns)
    datasets = [get_tf_dataset(folder + fn) for fn in train_fns]
    test_datasets = [get_tf_dataset(folder + fn) for fn in test_fns]
    return datasets, test_datasets


def train_directional_predictor(datasets, test_datasets):
    # EPOCHS = 10
    BATCH_SIZE = 1
    BUFFER_SIZE = 100

    datasets, test_datasets = get_directional_datasets()

    model, loss_object, optimizer = model_core()

    train_summary_writer, test_summary_writer = init_summary_writers()

    for epoch, dataset in enumerate(datasets):
        if not dataset:
            continue
        for (x_train, y_train) in dataset:
            tl, ta, correct = train_step(
                model, optimizer, loss_object, x_train, y_train
            )
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
        for (x_test, y_test) in test_dataset:
            tel, tea = test_step(model, loss_object, x_test, y_test)

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


def get_pred_df(df, cols=COLS, to_numpy=True):

    raw = df[cols]
    test_cols = ['a_team', 'h_team', 'status']  # order matters
    teams_map, statuses_map = h.dicts_for_one_hotting(
        sports=['nba', 'nfl', 'nhl'])
    hot_df = s.hot(raw, columns=test_cols, hot_maps=[
                   teams_map, teams_map, statuses_map])
    vals = {'EVEN': 100, 'None': -1, None: -1}
    full_df = hot_df.replace(vals)
    serialized = full_df.astype(np.float32)
    if to_numpy:
        serialized = serialized.values
    return serialized


def prep_pred_df(dataset):
    BATCH_SIZE = 1
    BUFFER_SIZE = 1
    past_history = 2
    future_target = 1
    STEP = 1
    EVALUATION_INTERVAL = 200
    EPOCHS = 10
    X, y = h.multivariate_data(dataset, dataset[:, 8:10], 0,
                               len(dataset) -
                               1, past_history,
                               future_target, STEP,
                               single_step=True)
    # X = tf.keras.utils.normalize(X)
    X = tf.data.Dataset.from_tensor_slices((X, y))
    # X = X.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return X


def df_to_tf_dataset(df):
    serialized = get_pred_df(df)
    X = prep_pred_df(serialized)
    return X


def get_dfs():
    folder = m.PROJ_DIR + "ml/lines/"
    fns = h.get_fns(folder)
    dfs = [pd.read_csv(folder + fn) for fn in fns]
    return dfs


def get_datasets():
    dfs = get_dfs()
    datasets = [df_to_tf_dataset(df) for df in dfs]
    return datasets


def main():
    datasets = get_datasets()
    loss_object = tf.losses.MeanAbsoluteError()
    for d in datasets:
        print(d)
        print(dir(d))
        break
    optimizer = tf.keras.optimizers.RMSprop(clipvalue=1.0)
    model = TfLSTM(in_shape=datasets[0][0].shape[-2:])
    for epoch, dataset in enumerate(datasets):
        if not dataset:
            continue
        for (x_train, y_train) in dataset:
            tl, ta, correct = train_step(
                model, optimizer, loss_object, x_train, y_train
            )


if __name__ == "__main__":
    # main()

    main()
