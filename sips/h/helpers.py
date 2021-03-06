"""

"""
import random

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from sips.macros import macros as m

from sips.h import fileio as fio


def get_dfs(to_read=None, dict_key=None, output="list"):
    """
    to_read is one of:
        - list of *full* file names 
        - path to folder 
    """
    if not to_read:
        to_read = m.PARENT_DIR + "data/lines/lines/"

    if isinstance(to_read, str):
        to_read = fio.get_fns(to_read)

    if output == "list":
        dfs = [pd.read_csv(fn) for fn in to_read]
    elif output == "dict":
        dfs = {}
        if not dict_key:
            dict_key = "game_id"
        for fn in to_read:
            df = pd.read_csv(fn)
            try:
                key = df[dict_key].iloc[0]
            except KeyError:
                continue

            dfs[key] = df

    return dfs


def seq_windows(
    dataset,
    target=None,
    start_index=0,
    end_index=None,
    history_size=1,
    target_size=1,
    step=1,
    single_step=False,
):
    """
    create sliding window tuples for training nns on multivar timeseries
    """
    data = []
    labels = []
    start_index = start_index + history_size

    if target is None:
        target = dataset

    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        X = dataset[indices]
        data.append(X)

        if single_step:
            label = target[i + target_size]
        else:
            label = target[i: i + target_size]

        labels.append(label)

    return np.array(data), np.array(labels)


def seq_windows_df(
    df,
    target=None,
    start_index=0,
    end_index=None,
    history_size=1,
    target_size=1,
    step=1,
    single_step=False,
):
    """
    create sliding window tuples for training nns on multivar timeseries
    """
    data = []
    labels = []
    start_index = start_index + history_size

    if target is None:
        target = df

    if end_index is None:
        end_index = df.shape[0] - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        X = df.iloc[indices]
        data.append(X)

        if single_step:
            label = target[i + target_size]
        else:
            label = target[i: i + target_size]

        labels.append(label)

    return data, labels


def train_test_split_list(to_split, train_frac=0.7, shuffle=False):
    """

    """
    length = len(to_split)
    split_idx = round(train_frac * length)

    if shuffle:
        random.shuffle(to_split)

    train_fns = to_split[0:split_idx]
    test_fns = to_split[split_idx:]

    return train_fns, test_fns


def split_by(df, by='game_id', train_frac=0.7):
    g_ids = df[by].unique()
    random.shuffle(g_ids)

    n = len(g_ids)
    idx = int(n * train_frac)

    tr_ids = g_ids[:idx]
    te_ids = g_ids[idx:]

    tr_df = df[df[by].isin(tr_ids)]
    te_df = df[df[by].isin(te_ids)]
    return tr_df, te_df


def remove_string_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_remove = []
    for col in df.columns:
        try:
            _ = df[col].astype(float)
        except ValueError:
            cols_to_remove.append(col)
            pass
    # keep only the columns in df that do not contain string
    df = df[[col for col in df.columns if col not in cols_to_remove]]
    return df


def chunk(df: pd.DataFrame, cols=["game_id"], output="list"):
    # returns a python dict of dfs, splitting the df arg by unique col value
    # df type pd df, col type string
    if output == "list":
        games = [game[1] for game in df.groupby(cols)]
    elif output == "dict":
        games = {key: val for key, val in df.groupby(cols)}
    elif output == 'df':
        games = {key: val for key, val in df.groupby(cols)}
        games = pd.DataFrame.from_dict(games, orient='index')
        games = games.iloc[:, 0]
    else:
        games = df.groupby(cols)
    return games


def apply_length_bounds(
    games, min_lines=200, max_lines=5000, output="list", verbose=False
):
    """
    given dict of game dataframes 
    and an integer > 0 for the minimum length of a game in csv lines
    """
    pre_len = len(games)
    deleted_dict = {}

    if isinstance(games, list):
        games = {i: game for i, game in enumerate(games)}

    for key, value in games.copy().items():
        game_len = value.shape[0]  # len(value)
        if game_len < min_lines and game_len < max_lines:
            deleted_dict[key] = game_len
            del games[key]

    if verbose:
        print(f"applied minimum game len of : {min_lines}\n")
        print(f"before apply: {pre_len}")
        print(f"after apply: {len(games)}\n")

    if output == "list":
        games = list(games.values())

    return games


def filter_unended(dfs, verbose=False) -> list:
    # filters dfs, removing df.iloc[-1].status != "GAME_END"
    full_games = []
    total_count = len(dfs)
    skips = 0
    for df in dfs:
        if df.iloc[-1].status == "GAME_END":
            full_games.append(df)
        else:
            skips += 1
    if verbose:
        print(f"filtered {skips} unended games out of {total_count}\n")

    return full_games


def labels_split(df: pd.DataFrame, cols, drop=True):
    # split df into X and Y, with option to drop Y from X
    X = df
    Y = X[[cols]].copy()
    if drop:
        X = df.drop(Y, axis=1)
    return X, Y


def sk_scale(df: pd.DataFrame, to_df=False):
    """
    scales pandas or np data(frame) using StandardScaler 
    returns numpy or dataframe (to_df=True)
    """
    scaler = StandardScaler()
    if isinstance(df, pd.core.frame.DataFrame):  # pandas
        cols = df.columns
        df = df.to_numpy()

    scaled = scaler.fit_transform(df)

    if to_df:
        scaled = pd.DataFrame(scaled, columns=cols)

    return scaled


def filter_then_apply_min(dfs, verbose=False) -> list:
    dfs = filter_unended(dfs, verbose=verbose)
    dfs = apply_length_bounds(dfs, verbose=verbose)
    return dfs


def apply_min_then_filter(dfs, min_lines=200, max_lines=5000, verbose=False) -> list:
    # faster than filter and apply
    dfs = apply_length_bounds(dfs,  min_lines=200, max_lines=5000, verbose=verbose)
    dfs = filter_unended(dfs, verbose=verbose)
    return dfs


def get_full_games(folder: str = None, sport: str = None) -> list:
    # sport: BASK, FOOT, etc
    dfs = get_dfs(folder)
    dfs = filter_then_apply_min(dfs)
    if sport is not None:
        dfs = filter_sport(dfs, sport)
    return dfs


def filter_sport(dfs: list, sport: str) -> list:
    return [df for df in dfs if df.sport[0] == sport]


if __name__ == "__main__":

    pass
