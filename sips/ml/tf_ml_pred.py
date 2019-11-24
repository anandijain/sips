import random
import datetime

import tensorflow as tf

from sips.h import helpers as h

from sips.macros import bov as bm
from sips.macros import macros as m
from sips.macros import tfm

from sips.ml import tf_loaders as tfls
from sips.ml import tf_utils as tfu
from sips.ml import tf_fwd
from sips.ml import tf_lstm as lstm


def ml_direction_predict():
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = tfm.WRITE_TO + "gradient_tape/" + current_time

    cols = bm.TO_SERIALIZE

    PRINT_INTERVAL = 30
    HISTORY_SIZE = 30
    PRED_SIZE = 30

    all_datasets = tfls.prediction_data_from_folder(
        tfm.READ_FROM,
        in_cols=cols,
        label_cols=["a_ml", "h_ml"],
        batch_size=1,
        buffer_size=1,
        history_size=HISTORY_SIZE,
        pred_size=PRED_SIZE,
        step_size=1,
        norm=True,
    )

    print(len(all_datasets))
    datasets, test_datasets = h.train_test_split_list(all_datasets)
    x, y = tfu.get_example(datasets)

    loss_fxn = tf.losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.RMSprop(clipvalue=1.0)
    model = lstm.make_model_functional(x.shape[-2:], tf.size(y[0]))

    train_loss, test_loss = tfu.get_loss_metrics()

    train_summary_writer, test_summary_writer = tfu.init_summary_writers(log_dir)
    print(f"log_dir: {log_dir}")

    train_step_num = 0
    test_step_num = 0
    for epoch, dataset in enumerate(datasets):
        if not dataset:
            continue

        for i, (x_train, y_train) in enumerate(dataset):
            tl, loss, predictions = tf_fwd.train_step_regress(
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

            if i % PRINT_INTERVAL == 0:
                print(
                    f"{i}: x_train: {x_train[-1][0:20]} \
                        preds:\
                        {tf.reshape(predictions, (-1, 2))}\
                        actuals:\
                        {tf.reshape(y_train, (-1, 2))}\
                        loss: {loss.numpy()}"
                )

        test_dataset = random.choice(test_datasets)

        if not test_dataset:
            continue

        for xte, yte in test_dataset:
            tel = tf_fwd.test_step(
                model, loss_fxn, xte, tf.reshape(yte, (1, -1)), test_loss
            )
            test_step_num += 1

            with test_summary_writer.as_default():
                tf.summary.scalar("loss", tel.numpy(), step=test_step_num)

        if epoch % 2000:
            template = "Epoch {}, Loss: {}, Test Loss: {}"
            print(template.format(epoch + 1, train_loss.result(), test_loss.result(),))

        # Reset metrics every epoch
        train_loss.reset_states()
        test_loss.reset_states()

    model_fn = (
        tfm.WRITE_TO + "models/ml_pred/" + str(HISTORY_SIZE) + "_" + str(PRED_SIZE)
    )

    tf.saved_model.save(model, model_fn)


if __name__ == "__main__":
    ml_direction_predict()
