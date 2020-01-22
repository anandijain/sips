import datetime

import tensorflow as tf

from sips.ml import lstm
from sips.macros import tfm


def get_example(datasets, verbose=False):
    """
    return an example of input to neural net (features)
        and an example of labels (ideal output of nn) 

    """
    x_train, y_train = None, None
    len_datasets = len(datasets)
    print(f"len_datasets: {len_datasets}")
    for i, dataset in enumerate(datasets):
        print(f"spec: {dataset.element_spec}")

        if not dataset:
            print(f"skipped: {i}")
            continue

        for example in dataset:
            x_train, y_train = example[0], example[1]
            print(f"x_train: {x_train}")
            print(f"x_train.shape: {x_train.shape}")
            print(f"y_train: {y_train}")
            print(f"y_train.shape: {y_train.shape}")
            break

    if verbose:
        print(f"x_train: {x_train}")
        print(f"y_train: {y_train}")
    return x_train, y_train


def classify_model_core(in_shape, out_dim):
    """
    get basic components of a classification model given specified in/out dim

    """
    model = lstm.make_model_seq(in_shape, out_dim)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    return model, loss_object, optimizer


def init_summary_writers(log_dir):
    """
    instantiate tf summary writers for tensorboard

    """
    train_log_dir = log_dir + "/train"
    test_log_dir = log_dir + "/test"

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    return train_summary_writer, test_summary_writer


def get_loss_metrics():
    """
    for logging loss over time

    """
    train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean("test_loss", dtype=tf.float32)
    return train_loss, test_loss


def get_acc_metrics():
    """
    for logging accuracy over time

    """
    train_accuracy = tf.keras.metrics.CategoricalAccuracy("train_accuracy")
    test_accuracy = tf.keras.metrics.CategoricalAccuracy("test_accuracy")
    return train_accuracy, test_accuracy


def get_classification_metrics():
    """
    classification metrics: loss and accuracy

    """
    train_loss, test_loss = get_loss_metrics()
    train_accuracy, test_accuracy = get_acc_metrics()
    return train_loss, train_accuracy, test_loss, test_accuracy


def model_save_fn(history_size, pred_size, ext=".pb"):
    """
    return the path for where to save the model

    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = (
        tfm.WRITE_TO
        + "models/"
        + str(history_size)
        + "_"
        + str(pred_size)
        + "_"
        + current_time
        + ext
    )
    return file_path


def get_logdir():
    """
    return path to tensorboard logdir

    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = tfm.WRITE_TO + "gradient_tape/" + current_time
    print(f"log_dir: {log_dir}")
    return log_dir
