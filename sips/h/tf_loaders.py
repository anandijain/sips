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

COLS = [
    # "num_markets",
    # "live",
    'a_team',
    'h_team',
    "quarter",
    "secs",
    "a_pts",
    "h_pts",
    "status",
    "a_ps",
    "h_ps",
    "a_hcap",
    "h_hcap",
    "a_ml",
    "h_ml",
    # "game_start_time",
]


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


def get_directional_datasets():
    folder = m.PROJ_DIR + "ml/lines/"
    fns = fio.get_fns(folder)
    train_fns, test_fns = fio.train_test_split_dir(fns)
    datasets = [get_tf_dataset(folder + fn) for fn in train_fns]
    test_datasets = [get_tf_dataset(folder + fn) for fn in test_fns]
    return datasets, test_datasets


def get_pred_df(df, cols=COLS, to_numpy=True):
    raw = df[cols]
    test_cols = ['a_team', 'h_team', 'status']  # order matters
    teams_map, statuses_map = h.dicts_for_one_hotting(
        sports=['nba', 'nfl', 'nhl'])
    hot_df = hot.hot(raw, columns=test_cols, hot_maps=[
        teams_map, teams_map, statuses_map])
    vals = {'EVEN': 100, 'None': -1, None: -1}
    full_df = hot_df.replace(vals)
    serialized = full_df.astype(np.float32)
    if to_numpy:
        serialized = serialized.values
    return serialized


def prep_pred_df(dataset):
    BATCH_SIZE = 1
    BUFFER_SIZE = 1
    past_history = 2
    future_target = 1
    STEP = 1
    EVALUATION_INTERVAL = 200
    EPOCHS = 10
    X, y = h.multivariate_data(dataset, dataset[:, 8:10], 0,
                               len(dataset) -
                               1, past_history,
                               future_target, STEP,
                               single_step=True)
    # X = tf.keras.utils.normalize(X)
    X = tf.data.Dataset.from_tensor_slices((X, y))
    # X = X.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return X


def df_to_tf_dataset(df):
    serialized = get_pred_df(df)
    X = prep_pred_df(serialized)
    return X


def get_pred_datasets(folder):
    dfs = h.get_dfs(folder)
    datasets = [df_to_tf_dataset(df) for df in dfs]
    return datasets
