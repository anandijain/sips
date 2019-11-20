import os
import datetime
import random

import pandas as pd
import numpy as np
import tensorflow as tf

from sips.macros import macros as m
from sips.macros import bov as bm
from sips.lines.bov import bov
from sips.h import helpers as h
from sips.h import serialize as s


class TfLSTM(tf.keras.Model):
    """
    subclassing model type
    """

    def __init__(self):
        super(TfLSTM, self).__init__()
        self.l1 = tf.keras.layers.LSTM(100, activation="relu")
        self.l2 = tf.keras.layers.LSTM(128, activation="relu")
        self.l3 = tf.keras.layers.Dense(19, activation="softmax")

    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


def multivariate_data(
    dataset,
    target,
    start_index,
    end_index,
    history_size,
    target_size,
    step,
    single_step=False,
):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i : i + target_size])

    return np.array(data), np.array(labels)


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

    past_history = 15
    future_target = 5
    STEP = 1

    # x_train_single, y_train_single = multivariate_data(X, y, 0,
    #                                                 len(y), past_history,
    #                                                 future_target, STEP,
    #                                                 single_step=False)
    # x_train_single = tf.reshape(x_train_single, ())
    # print('Single window of past history : {}'.format(x_train_single[0]))
    # print('Single window of past history shape: {}'.format(x_train_single[0].shape))

    # print('Single window of past history : {}'.format(y_train_single[0]))
    # print('Single window of past history shape: {}'.format(y_train_single[0].shape))
    # dataset = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))

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


def get_fns(dir):
    fns = os.listdir(dir)
    try:
        fns.remove("LOG.csv")
    except ValueError:
        pass

    return fns


def train_test_split_dir(fns, train_frac=0.7, shuffle=False):
    """

    """
    num_files = len(fns)
    split_idx = round(0.7 * num_files)

    if shuffle:
        random.shuffle(fns)

    train_fns = fns[0:split_idx]
    test_fns = fns[split_idx:]

    return train_fns, test_fns


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


def main():
    # EPOCHS = 10
    BATCH_SIZE = 1
    BUFFER_SIZE = 10000

    folder = "./lines/"
    fns = get_fns(folder)
    
    train_fns, test_fns = train_test_split_dir(fns)

    datasets = [get_tf_dataset(folder + fn) for fn in train_fns]
    test_datasets = [get_tf_dataset(folder + fn) for fn in test_fns]

    # single dataset example
    dataset = datasets[0]
    test_dataset = test_datasets[0]

    # model = TfLSTM()
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


def get_pred_df():
    cols = [
        "num_markets",
        "live",
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
        "game_start_time",
    ]
    folder = m.PROJ_DIR + "ml/lines/"
    fns = get_fns(folder)
    df = pd.read_csv(folder + fns[0])

    raw = df[cols]
    test_cols = ['a_team', 'h_team', 'status']  # order matters
    teams_map, statuses_map = h.dicts_for_one_hotting(sports=['nba', 'nfl', 'nhl']) 
    hot_df = s.hot(df, columns=test_cols, hot_maps=[teams_map, teams_map, statuses_map])

    return hot_df

if __name__ == "__main__":
    # main()
    hot_df = get_pred_df()
    print(f'hot_df: {hot_df}')