import os
import datetime
import random

import pandas as pd
import numpy as np
import tensorflow as tf


from sips.lines.bov import bov


class TfLSTM(tf.keras.Model):
    '''
    subclassing model type
    '''

    def __init__(self):
        super(TfLSTM, self).__init__()
        self.l1 = tf.keras.layers.Dense(100, activation='relu')
        self.l2 = tf.keras.layers.Dense(128, activation='relu')
        self.l3 = tf.keras.layers.Dense(19, activation='softmax')

    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


def make_model():
    # sequential model

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(19, activation='softmax')
    ])
    return model


def get_tf_dataset(fn):
    data = bov.prep_game_dataset(fn)
    X, y = data
    print(f'X: {X}, X[0].shape: {X[0].shape}')
    print(f'y: {y}')
    # tf_X = tf.convert_to_tensor(X)
    # tf_y = tf.convert_to_tensor(y)

    dataset = tf.data.Dataset.from_tensor_slices((np.array(X), np.array(y)))

    return dataset


train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    'train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')



def train_step(model, optimizer, loss_object, x_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
        print(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    tl = train_loss(loss)
    ta = train_accuracy(y_train, predictions)
    return tl, ta


def test_step(model, loss_object, x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)

    te = test_loss(loss)
    tea = test_accuracy(y_test, predictions)
    return te, tea

def main():
    # EPOCHS = 10
    folder = './lines/'
    fns = os.listdir(folder)
    fns.remove('LOG.csv')
    num_files = len(fns)
    train_frac = 0.7
    split_idx = round(0.7 * num_files)
    random.shuffle(fns)
    train_fns = fns[0:split_idx]
    test_fns = fns[split_idx:]

    datasets = [get_tf_dataset(folder + fn) for fn in train_fns]
    test_datasets = [get_tf_dataset(folder + fn) for fn in test_fns]

    # model = TfLSTM()
    model = make_model()
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()



    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)


    for epoch, dataset in enumerate(datasets):
        for (x_train, y_train) in dataset:
            tl, ta = train_step(model, optimizer, loss_object, x_train, y_train)
        with train_summary_writer.as_default():
            
            tf.summary.scalar('loss', tl.numpy(), step=epoch)
            tf.summary.scalar('accuracy', ta.numpy(), step=epoch)

        test_dataset = random.choice(test_datasets)
        for (x_test, y_test) in test_dataset:
            tel, tea = test_step(model, loss_object, x_test, y_test)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', tel.numpy(), step=epoch)
            tf.summary.scalar('accuracy', tea.numpy(), step=epoch)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch+1,
                                train_loss.result(),
                                train_accuracy.result()*100,
                                test_loss.result(),
                                test_accuracy.result()*100))

        # Reset metrics every epoch
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()





if __name__ == "__main__":
    main()
