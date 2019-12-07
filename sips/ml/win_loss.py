import random
import tensorflow as tf

import sips.macros.bov as bm
from sips.macros import tfm

import sips.h.helpers as h
import sips.h.serialize as s
import sips.h.attach as a

from sips.ml import fwd
from sips.ml import lstm

import sips.ml.loading as tfls
from sips.ml import utils as tfu


def get_wl_datasets():
    """

    """
    dfs = h.get_dfs()
    dfs_w_win = a.wins(dfs)

    sXs, sYs = s.serialize_dfs(
        dfs_w_win,
        in_cols=bm.TO_SERIALIZE,
        label_cols=["a_win", "h_win"],
        drop_labels=True,
        norm=True,
    )

    all_datasets = tfls.serialized_to_datasets(
        sXs, sYs, history_size=1, pred_size=1, single_step=True
    )
    datasets, test_datasets = h.train_test_split_list(all_datasets)
    return datasets, test_datasets


def wl_predict(datasets, test_datasets):
    """

    """
    log_dir = tfu.get_logdir()
    datasets, test_datasets = get_wl_datasets()

    x, y = tfu.get_example(datasets)

    loss_fxn = tf.losses.BinaryCrossentropy()
    optim = tf.keras.optimizers.Adam()
    model = lstm.make_mlp_functional(x.shape[-2:], tf.size(y[0]), classify=True)

    train_summary_writer, test_summary_writer = tfu.init_summary_writers(log_dir)

    (
        train_loss,
        train_accuracy,
        test_loss,
        test_accuracy,
    ) = tfu.get_classification_metrics()

    tr_step = 0
    te_step = 0

    for epoch, dataset in enumerate(datasets):
        if not dataset:
            continue
        for (xtr, ytr) in dataset:
            ytr = tf.reshape(ytr, (1, -1))
            tl, ta, preds = fwd.train_step_classify(
                model, optim, loss_fxn, xtr, ytr, train_loss, train_accuracy
            )
            tr_step += 1

            with train_summary_writer.as_default():
                tf.summary.scalar("loss", tl.numpy(), step=tr_step)
                tf.summary.scalar("accuracy", ta.numpy(), step=tr_step)

            maxed_pred = tf.argmax(preds, 1).numpy()[0]
            maxed_true = tf.argmax(ytr).numpy()
            correct = tf.equal(maxed_pred, maxed_true).numpy()

        test_dataset = random.choice(test_datasets)

        if not test_dataset:
            continue
        for (xte, yte) in test_dataset:
            yte = tf.reshape(yte, (1, -1))

            tel, tea = fwd.test_step(
                model, loss_fxn, xte, yte, test_loss, test_accuracy
            )
            te_step += 1

            with test_summary_writer.as_default():
                tf.summary.scalar("loss", tel.numpy(), step=te_step)
                tf.summary.scalar("accuracy", tea.numpy(), step=te_step)

        template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, Preds: {}, Acts: {}"
        print(
            template.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result() * 100,
                test_loss.result(),
                test_accuracy.result() * 100,
                preds,
                ytr,
            )
        )

        # Reset metrics every epoch
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

    tf.saved_model.save(model, tfm.WRITE_TO + "win_loss/")

    return datasets


def main():
    """

    """
    train, test = get_wl_datasets()
    wl_predict(train, test)


if __name__ == "__main__":
    main()
