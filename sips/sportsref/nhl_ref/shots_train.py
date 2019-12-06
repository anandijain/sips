import pandas as pd
import tensorflow as tf
import numpy as np


def to_cat_codes(df, col):
    """

    """
    df[col] = pd.Categorical(df[col])
    return df[col].cat.codes


def convert_cols(df, columns):
    """

    """
    for col in columns:
        df[col] = to_cat_codes(df, col)
    return df


def parse_shots_df(df):
    """

    """
    cols = ["type", "outcome", "player", "game_id"]
    df = convert_cols(df, cols)

    df["target"] = df.outcome
    df = df.drop(columns=["i", "type", "outcome"])
    df = df.astype(np.uint16)
    print(df)
    return df


def get_compiled_model():
    """

    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(20, activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def datasets():
    """

    """
    train = pd.read_csv("./data/test.csv")
    test = pd.read_csv("./data/shots.csv")

    train = parse_shots_df(train)
    test = parse_shots_df(test)

    train_labels = train.pop("target")
    test_labels = test.pop("target")

    train_ds = tf.data.Dataset.from_tensor_slices((train.values, train_labels.values))
    test_ds = tf.data.Dataset.from_tensor_slices((test.values, test_labels.values))

    train_dataset = train_ds.shuffle(len(train)).batch(128)
    test_dataset = test_ds.shuffle(len(test)).batch(128)

    # print(train, test)
    return train_dataset, train_dataset


def main():
    """

    """
    fn = "./data/model1/test"
    train, test = datasets()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=fn, save_weights_only=True, verbose=1
    )
    # return train, test
    try:
        model = tf.load(fn)
    except:
        model = get_compiled_model()
        pass
    model.fit(train, epochs=5, callbacks=[cp_callback], validation_data=test)
    model.evaluate(test)
    model.save("./data/model1/test")
    model.summary()
    return model


if __name__ == "__main__":
    model = main()
