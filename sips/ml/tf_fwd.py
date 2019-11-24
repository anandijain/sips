import tensorflow as tf
from sips.macros import bov as bm


def train_step_classify(
    model, optimizer, loss_object, x_train, y_train, train_loss, train_accuracy, verbose=False
):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
        maxed_pred = tf.argmax(predictions, 1).numpy()[0]
        maxed_true = tf.argmax(y_train).numpy()
        # assumes batch size 1
        correct = tf.equal(maxed_pred, maxed_true).numpy()
        strs = bm.TRANSITION_CLASS_STRINGS

        if verbose:
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


def train_step_regress(model, optimizer, loss_object, x_train, y_train, train_loss):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    tl = train_loss(loss)
    return tl, loss, predictions


def test_step(model, loss_object, x_test, y_test, test_loss, test_accuracy=None):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)

    te_loss = test_loss(loss)
    if test_accuracy:
        tea = test_accuracy(y_test, predictions)
        return te_loss, tea
    else:
        return te_loss
