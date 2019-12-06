import random

import tensorflow as tf

from sips.h import helpers as h

from sips.macros import bov as bm
from sips.macros import macros as m
from sips.macros import tfm

from sips.ml import loading as tfls
from sips.ml import fwd
from sips.ml import utils as tfu
from sips.ml import lstm


def ml_predict(datasets, test_datasets, log_dir, model_fn, num_epochs=1):

    weights_path = tfu.model_save_fn(HISTORY_SIZE, PRED_SIZE, ext="tf")

    x, y = tfu.get_example(datasets, verbose=True)

    loss_fxn = tf.losses.MeanAbsolutePercentageError()
    optimizer = tf.keras.optimizers.RMSprop()
    model = lstm.make_lstm_functional(x.shape[-2:], tf.size(y[0]))

    train_loss, test_loss = tfu.get_loss_metrics()

    train_summary_writer, test_summary_writer = tfu.init_summary_writers(log_dir)
    print(f"log_dir: {log_dir}")

    train_step_num = 0
    test_step_num = 0
    for epoch in range(NUM_EPOCHS):
        for game_num, dataset in enumerate(datasets):
            if not dataset:
                continue

            for i, (x_train, y_train) in enumerate(dataset):
                tl, loss, predictions = fwd.train_step_regress(
                    model,
                    optimizer,
                    loss_fxn,
                    x_train,
                    tf.reshape(y_train, (1, -1)),
                    train_loss,
                )
                train_step_num += 1

                with train_summary_writer.as_default():
                    tf.summary.scalar("loss", tl.numpy(), step=train_step_num)

                train_loss.reset_states()

                if i % PRINT_INTERVAL == 0:
                    print(
                        f"{i}\n "
                        "preds:\n"
                        f"{tf.reshape(predictions, (-1, 2))}\n"
                        f"actuals:\n"
                        f"{tf.reshape(y_train, (-1, 2))}"
                        f"loss: {loss.numpy()}"
                    )
            model.reset_states()

            test_dataset = random.choice(test_datasets)

            if not test_dataset:
                continue

            for xte, yte in test_dataset:
                tel = fwd.test_step(
                    model, loss_fxn, xte, tf.reshape(yte, (1, -1)), test_loss
                )
                test_step_num += 1

                with test_summary_writer.as_default():
                    tf.summary.scalar("loss", tel.numpy(), step=test_step_num)

                test_loss.reset_states()

            if game_num % 2000:
                template = "game_num {}, Loss: {}, Test Loss: {}"
                print(
                    template.format(
                        game_num + 1, train_loss.result(), test_loss.result(),
                    )
                )

            # Reset metrics every epoch
            model.reset_states()

        tf.saved_model.save(model, model_fn)
        model.save_weights(weights_path, save_format="tf")


def get_datasets(history_size, pred_size):

    all_datasets = tfls.prediction_data_from_folder(
        folder=tfm.READ_FROM,
        in_cols=None,
        label_cols=["a_ml", "h_ml"],
        batch_size=1,
        buffer_size=1,
        history_size=HISTORY_SIZE,
        pred_size=PRED_SIZE,
        step_size=1,
        norm=True,
        verbose=True,
    )

    print(f"num datasets: {len(all_datasets)}")
    datasets, test_datasets = h.train_test_split_list(all_datasets)
    return datasets, test_datasets


def main(history_size, pred_size, num_epochs=1):

    datasets, test_datasets = get_datasets(HISTORY_SIZE, PRED_SIZE)
    model_fn = tfu.model_save_fn(HISTORY_SIZE, PRED_SIZE)
    log_dir = tfu.get_logdir()

    ml_predict(datasets, test_datasets, log_dir, model_fn, num_epochs=NUM_EPOCHS)


if __name__ == "__main__":
    HISTORY_SIZE = 1
    PRED_SIZE = 30
    PRINT_INTERVAL = PRED_SIZE
    NUM_EPOCHS = 2

    main(HISTORY_SIZE, PRED_SIZE, num_epochs=NUM_EPOCHS)
