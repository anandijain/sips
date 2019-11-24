import tensorflow as tf


class TfLSTM(tf.keras.Model):
    """
    subclassing model type
    """

    def __init__(self, in_dim):
        super(TfLSTM, self).__init__()
        # self.e1 = tf.keras.layers.Embedding(input_dim=in_shape, output_dim=64)
        self.l1 = tf.keras.layers.LSTM(
            100, activation="relu", input_shape=(None, in_dim), return_sequences=True
        )
        self.l2 = tf.keras.layers.LSTM(128, activation="relu")
        self.l3 = tf.keras.layers.Dense(19, activation="softmax")

    def call(self, x):
        print(x.shape)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


def make_model_seq(in_shape, out_dim):
    # sequential model

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.LSTM(
                100, input_shape=in_shape, return_sequences=True, activation="relu"
            ),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(out_dim, activation="softmax"),
        ]
    )
    return model


def make_model_functional(in_shape_tup, out_dim):
    inputs = tf.keras.Input(shape=in_shape_tup, batch_size=1)
    x = tf.keras.layers.LSTM(200, activation="relu")(inputs)
    x = tf.keras.layers.LSTM(128, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(100)(x)
    outputs = tf.keras.layers.Dense(out_dim)(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def main():
    """

    """
    model = make_model_functional((30, 200), (1, 60))
    return model


if __name__ == "__main__":
    # main()
    model = main()
    print(model)
