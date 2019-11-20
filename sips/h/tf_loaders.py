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

FOLDER = m.PROJ_DIR + "ml/lines/"


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


def datasets_from_dir(get_dataset_fxn, folder=FOLDER, train_frac=0.7):
    '''
    trying to generalize usage so that i can apply a function that prepares
    a folder

    not working yet
    '''
    fns = fio.get_fns(folder)
    print(fns)
    dfs = h.get_dfs(fns)
    train_fns, test_fns = h.train_test_split_list(dfs, train_frac=train_frac)
    datasets = [get_dataset_fxn(folder + fn) for fn in train_fns]
    test_datasets = [get_dataset_fxn(folder + fn) for fn in test_fns]
    return datasets, test_datasets



def get_pred_df(df, cols=COLS, to_numpy=True):
    """

    """
    raw = df[cols]
    test_cols = ['a_team', 'h_team', 'status']  # order matters
    teams_map, statuses_map = hot.dicts_for_one_hotting(
        sports=['nba', 'nfl', 'nhl'])
    hot_df = hot.hot(raw, columns=test_cols, hot_maps=[
        teams_map, teams_map, statuses_map])
    vals = {'EVEN': 100, 'None': -1, None: -1}
    full_df = hot_df.replace(vals)
    serialized = full_df.astype(np.float32)
    if to_numpy:
        serialized = serialized.values
    return serialized


def prep_pred_df(dataset, batch_size=1, buffer_size=1, history_size=10, pred_size=1, step_size=1, norm=True):
    X, y = h.multivariate_data(dataset, dataset[:, 8:10], 0,
                               len(dataset) -
                               1, history_size,
                               pred_size, step_size,
                               single_step=False)
    if norm:
        X = tf.keras.utils.normalize(X)

    X = tf.data.Dataset.from_tensor_slices((X, y))
    # X = X.cache().shuffle(buffer_size).batch(batch_size)
    X = X.batch(batch_size)
    # X = X.take(batch_size).shuffle(buffer_size).cache().repeat()
    return X


def df_to_tf_dataset(df, batch_size=1, buffer_size=1, history_size=10, pred_size=1, step_size=1, norm=True):
    serialized = get_pred_df(df)
    X = prep_pred_df(serialized, batch_size, buffer_size,
                     history_size, pred_size, step_size, norm)
    return X


def get_pred_datasets(folder, label_cols, batch_size=1, buffer_size=1, history_size=10, pred_size=1, step_size=1, norm=True):
    dfs = h.get_dfs(folder)
    datasets = [df_to_tf_dataset(df, batch_size=1, buffer_size=1,
                                 history_size=10, pred_size=1, step_size=1, norm=True) for df in dfs]
    return datasets


def test_datasets_from_dir():
    """
    attempts to read folder and convert to train/test tf datasets lists 
    """
    datasets = datasets_from_dir(df_to_tf_dataset)
    return datasets


if __name__ == "__main__":
    datasets = get_pred_datasets(FOLDER, COLS)
