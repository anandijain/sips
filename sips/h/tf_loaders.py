import tensorflow as tf
import numpy as np
import pandas as pd

from sips.h import analyze
from sips.h import fileio as fio
from sips.h import hot
import sips.h.serialize as s
import sips.h.helpers as h


from sips.macros import bov as bm
from sips.macros import macros as m
import sips.h.calc as c


# COLS = [
#     # "num_markets",
#     # "live",
#     'a_team',
#     'h_team',
#     "quarter",
#     "secs",
#     "a_pts",
#     "h_pts",
#     "status",
#     "a_ps",
#     "h_ps",
#     "a_hcap",
#     "h_hcap",
#     "a_ml",
#     "h_ml",
#     # "game_start_time",
# ]

COLS = None

FOLDER = m.PARENT_DIR + "data/lines/lines"


def get_tf_dataset(fn, verbose=False):
    data = prep_game_dataset(fn)
    if not data:
        return
    X, y = data
    if verbose:
        print(f"X: {X}, X[0].shape: {X[0].shape}")
        print(f"y: {y}")
    # tf_X = tf.convert_to_tensor(X)
    # tf_y = tf.convert_to_tensor(y)
    X = tf.keras.utils.normalize(X)

    dataset = tf.data.Dataset.from_tensor_slices((np.array(X), np.array(y)))

    return dataset


def prep_game_dataset(fn, sports=["nba"]):  # , zip_data=True, verbose=False):
    teams_dict, statuses_dict = hot.dicts_for_one_hotting()

    df = pd.read_csv(fn)
    prev = [None, None]
    prev_row = [None for _ in range(25)]
    X = []
    y = []
    for i, row in df.iterrows():

        cur_row = row.values
        cur_ml = list(row[["a_ml", "h_ml"]])
        if i == 0:
            prev_ml = cur_ml
            prev_row = cur_row
            continue
        transition_class = analyze.classify_transition(prev_ml, cur_ml)
        if bm.TRANSITION_CLASS_STRINGS[np.argmax(transition_class)] == "stays same":
            continue

        x = s.serialize_row(prev_row, teams_dict, statuses_dict)
        y.append(transition_class)
        X.append(x)
        prev_ml = cur_ml
        prev_row = cur_row

    len_game = len(y)
    if not X:
        return
    X = np.reshape(np.concatenate(X, axis=0), (len_game, 1, -1))

    return X, y


def get_directional_datasets(train_frac=0.7):
    fns = fio.get_fns(FOLDER)
    train_fns, test_fns = h.train_test_split_list(fns, train_frac=train_frac)
    datasets = [get_tf_dataset(FOLDER + fn) for fn in train_fns]
    test_datasets = [get_tf_dataset(FOLDER + fn) for fn in test_fns]
    return datasets, test_datasets


def prediction_data_from_folder(
    folder=FOLDER,
    in_cols=None,
    label_cols=["a_ml", "h_ml"],
    batch_size=1,
    buffer_size=1,
    history_size=10,
    pred_size=2,
    step_size=1,
    norm=True,
):
    """

    """
    replace_data_map = {"None": np.nan, "EVEN": 100}
    hot_maps = hot.all_hot_maps(output="dict")
    dfs = h.get_dfs(folder)

    train_df_vals, train_df_labs = s.serialize_dfs(
        dfs,
        in_cols,
        label_cols,
        replace_dict=replace_data_map,
        hot_maps=hot_maps,
        norm=norm,
    )

    datasets = []
    for i in range(len(train_df_vals)):
        data = train_df_vals[i].astype(np.float32)
        length = len(data)
        if length < history_size + pred_size:
            continue
        targets = train_df_labs[i]
        X_windows, y_windows = h.multivariate_data(
            data, targets, history_size=history_size, target_size=pred_size
        )

        x_dataset = tf.data.Dataset.from_tensor_slices(X_windows)
        y_dataset = tf.data.Dataset.from_tensor_slices(y_windows)
        prepped = tf.data.Dataset.zip((x_dataset, y_dataset)).batch(batch_size)
        datasets.append(prepped)
    return datasets


# def test_datasets_from_dir():
#     """
#     attempts to read folder and convert to train/test tf datasets lists
#     """
#     datasets = datasets_from_dir(df_to_tf_dataset)
#     return datasets


if __name__ == "__main__":
    datasets = prediction_data_from_folder(FOLDER)
    print(len(datasets))
