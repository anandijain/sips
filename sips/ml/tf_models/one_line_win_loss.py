import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
from sips.ml import fwd

from sips.ml import utils as tfu
from sips.ml import lstm
import random
from sips.macros import tfm

DIR = m.DATA_DIR + 'nba/'
FILES = ["nba_history_with_stats.csv", "nba_history.csv"]
m.DATA_DIR + 'nba/nba_history_with_stats.csv"


class WL(Model):
    def __init__(self):
        super(WL, self).__init__()

        self.d1 = Dense(500, activation="relu")
        self.d2 = Dense(250, activation="relu")
        self.d3 = Dense(128, activation="relu")
        self.d4 = Dense(64, activation="relu")
        self.d5 = Dense(2, activation="softmax")

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return self.d5(x)


def load_data(batch_size=1):
    # m = m.select_dtypes(exclude=['object'])
    df, df2 = [pd.read_csv(DIR + f) for f in FILES]
    m = df.merge(df2, left_on="Game_id", right_on="Game_id", how="outer")
    m = m.dropna()
    m = m.apply(pd.to_numeric, errors="coerce")
    target = pd.get_dummies(m.pop("H_win"))
    target.columns = ["H_win", "A_win"]
    m = (m - m.mean()) / m.std()
    data = tf.data.Dataset.from_tensor_slices((m.values, target.values)).batch(
        batch_size
    )
    return data


# def main():
# datasets = [load_data()]
# x, y = tfu.get_example(datasets)
# loss_fxn = tf.losses.BinaryCrossentropy()
# optim = tf.keras.optimizers.Adam()
# model = lstm.make_mlp_functional(
#     238, 2, classify=True)
# log_dir = tfu.get_logdir()
# train_step = 0

# train_summary_writer, test_summary_writer = tfu.init_summary_writers(
#     log_dir)

# (
#     train_loss,
#     train_accuracy,
# ) = tfu.get_classification_metrics()

# for epoch, dataset in enumerate(datasets):
#     if not dataset:
#         continue
#     for i, (xtr, ytr) in enumerate(dataset):
#         ytr = tf.reshape(ytr, (1, -1))
#         xtr = tf.reshape(xtr, (1, -1))
#         train_loss, train_accuracy, preds = fwd.train_step_classify(
#             model, optim, loss_fxn, xtr, ytr, verbose=True
#         )
#         train_step += 1

#         maxed_pred = tf.argmax(preds, 1).numpy()[0]
#         maxed_true = tf.argmax(ytr).numpy()
#         correct = tf.equal(maxed_pred, maxed_true).numpy()
#         print(f'step: {i}, loss: {train_loss}, correct: {correct}, preds: {preds}')
#     template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Preds: {}, Acts: {}"
#     print(
#         template.format(
#             epoch + 1,
#             train_loss.result(),
#             train_accuracy.result() * 100,
#             preds,
#             ytr,
#         )
#     )

#     # Reset metrics every epoch
#     train_loss.reset_states()
#     train_accuracy.reset_states()

#     tf.saved_model.save(model, tfm.WRITE_TO + "win_loss/")

# return datasets


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(x, training=True)
        loss = loss_object(y, predictions)
        print(f"{loss},{dir(loss)}")
        print(predictions)
        print(y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y, predictions)


if __name__ == "__main__":
    data = load_data()
    model = WL()
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.BinaryCrossentropy(name="train_accuracy")

    EPOCHS = 1

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

        for x, y in data:
            train_step(x, y)

        print(f"{epoch+1}, {train_loss.result()}")
