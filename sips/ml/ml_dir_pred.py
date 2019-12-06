import random

import tensorflow as tf

import sips.h.helpers as h

from sips.macros import tfm
from sips.h import hot

from sips.ml import loading as tfls
from sips.ml import fwd
from sips.ml import utils as tfu
from sips.ml import lstm


def train_directional_predictor(datasets, test_datasets, NUM_EPOCHS=1):
    x, y = tfu.get_example(datasets)
    model, loss_fxn, optim = tfu.classify_model_core(x.shape[-2:], tf.size(y[0]))
    log_dir = tfu.get_logdir()

    train_summary_writer, test_summary_writer = tfu.init_summary_writers(log_dir)

    (
        train_loss,
        train_accuracy,
        test_loss,
        test_accuracy,
    ) = tfu.get_classification_metrics()

    train_step = 0
    test_step = 0
    for epoch in range(NUM_EPOCHS):
        for dataset_num, dataset in enumerate(datasets):
            if not dataset:
                continue
            for (xtr, ytr) in dataset:
                tl, ta, preds = fwd.train_step_classify(
                    model, optim, loss_fxn, xtr, ytr, train_loss, train_accuracy
                )
                train_step += 1

                with train_summary_writer.as_default():

                    tf.summary.scalar("loss", tl.numpy(), step=train_step)
                    tf.summary.scalar("accuracy", ta.numpy(), step=train_step)

                train_loss.reset_states()
                train_accuracy.reset_states()

            model.reset_states()
            test_dataset = random.choice(test_datasets)

            if not test_dataset:
                continue
            for (xte, yte) in test_dataset:
                tel, tea = fwd.test_step(
                    model, loss_fxn, xte, yte, test_loss, test_accuracy
                )
                test_step += 1

                with test_summary_writer.as_default():
                    tf.summary.scalar("loss", tel.numpy(), step=test_step)
                    tf.summary.scalar("accuracy", tea.numpy(), step=test_step)

                test_loss.reset_states()
                test_accuracy.reset_states()

            template = "GAME {}\n, Loss: {}\n, Accuracy: {}\n, Test Loss: {}\n, Test Accuracy: {},\n Preds: {}\n Actuals: {}\n"
            print(
                template.format(
                    dataset_num + 1,
                    train_loss.result(),
                    train_accuracy.result() * 100,
                    test_loss.result(),
                    test_accuracy.result() * 100,
                    preds,
                    ytr,
                )
            )
            model.reset_states()

        tf.saved_model.save(model, tfm.WRITE_TO + "directional_prediction/")


def main():

    hot_maps = hot.all_hot_maps()
    all_datasets = tfls.transition_datasets_from_folder(
        tfm.READ_FROM, hot_maps=hot_maps, single_step=True
    )
    datasets, test_datasets = h.train_test_split_list(all_datasets, shuffle=True)
    train_directional_predictor(datasets, test_datasets)


if __name__ == "__main__":
    main()
