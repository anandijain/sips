import datetime
import random
import tensorflow as tf

import sips.macros.bov as bm
from sips.macros import tfm

import sips.h.helpers as h
import sips.h.serialize as s
import sips.h.attach as a

from sips.ml import tf_fwd
import sips.ml.tf_loaders as tfl
import sips.ml.tf_utils as tfu
import sips.ml.tf_lstm as lstm


def get_wl_datasets():
    dfs = h.get_dfs()
    all_dfs_w_labels = []
    for df in dfs:
        if df is not None:
            df.drop(["a_ou", "h_ou"], axis=1, inplace=True)
            # print(df)
            with_labels = a.wins(df)
            all_dfs_w_labels.append(with_labels)

    sXs, sYs = s.serialize_dfs(
        all_dfs_w_labels,
        in_cols=bm.TO_SERIALIZE,
        label_cols=["a_win", "h_win"],
        drop_labs=True,
        norm=True
    )

    datasets = tfl.serialized_to_datasets(sXs, sYs, history_size=1, pred_size=1)
    return datasets


def main():
    all_datasets = get_wl_datasets()
    datasets, test_datasets = h.train_test_split_list(all_datasets)
    x, y = tfu.get_example(datasets)

    loss_fxn = tf.losses.BinaryCrossentropy()
    optim = tf.keras.optimizers.Adadelta()
    model = lstm.make_mlp_functional(x.shape[-2:], tf.size(y[0]), classify=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = tfm.WRITE_TO + "gradient_tape/" + current_time
    print(f"log_dir: {log_dir}")

    train_summary_writer, test_summary_writer = tfu.init_summary_writers(
        log_dir)

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
            tl, ta, preds = tf_fwd.train_step_classify(
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

            tel, tea = tf_fwd.test_step(
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
                ytr
            )
        )

        # Reset metrics every epoch
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

    tf.saved_model.save(model, tfm.WRITE_TO + "win_loss/")


    return datasets


if __name__ == "__main__":
    datasets = main()
    print(datasets[0].element_spec)
