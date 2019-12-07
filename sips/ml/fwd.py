import tensorflow as tf

from sips.macros import bov as bm


def train_step_classify(
    model,
    optimizer,
    loss_object,
    x_train,
    y_train,
    train_loss,
    train_accuracy,
    verbose=False,
):
    """

    """
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)

        if verbose:
            print(f"preds: {predictions}")
            print(f"actuals: {y_train}")
            print(loss.numpy())

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    tl = train_loss(loss)
    ta = train_accuracy(y_train, predictions)
    return tl, ta, predictions


def train_step_regress(model, optimizer, loss_object, x_train, y_train, train_loss):
    """

    """
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    tl = train_loss(loss)
    return tl, loss, predictions


def test_step(model, loss_object, x_test, y_test, test_loss, test_accuracy=None):
    """

    """
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)

    te_loss = test_loss(loss)
    if test_accuracy:
        tea = test_accuracy(y_test, predictions)
        return te_loss, tea
    else:
        return te_loss


if __name__ == "__main__":
    pass
