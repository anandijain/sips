import tensorflow as tf
import datetime
import random

from sips.macros import tfm

from sips.h import hot
import sips.h.helpers as h

from sips.ml import tf_loaders as tfls
from sips.ml import tf_fwd
from sips.ml import tf_utils as tfu
from sips.ml import tf_lstm as lstm


def train_directional_predictor(datasets, test_datasets):
    # EPOCHS = 10
    BATCH_SIZE = 1
    BUFFER_SIZE = 100
    x, y = tfu.get_example(datasets)

    model, loss_fxn, optim = tfu.classify_model_core(x.shape[-2:], tf.size(y[0]))

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = tfm.WRITE_TO + "gradient_tape/" + current_time
    print(f"log_dir: {log_dir}")

    train_summary_writer, test_summary_writer = tfu.init_summary_writers(log_dir)

    (
        train_loss,
        train_accuracy,
        test_loss,
        test_accuracy,
    ) = tfu.get_classification_metrics()
    
    for epoch, dataset in enumerate(datasets):
        if not dataset:
            continue
        for (xtr, ytr) in dataset:
            tl, ta, correct = tf_fwd.train_step_classify(
                model, optim, loss_fxn, xtr, ytr, train_loss, train_accuracy
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
        for (xte, yte) in test_dataset:
            tel, tea = tf_fwd.test_step(
                model, loss_fxn, xte, yte, test_loss, test_accuracy
            )

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

    tf.saved_model.save(model, tfm.WRITE_TO + "directional_prediction/")


def main():
    hot_maps = hot.all_hot_maps()
    all_datasets = tfls.transition_datasets_from_folder(
        tfm.READ_FROM, hot_maps=hot_maps
    )
    datasets, test_datasets = h.train_test_split_list(all_datasets, shuffle=True)
    train_directional_predictor(datasets, test_datasets)


if __name__ == "__main__":
    main()
