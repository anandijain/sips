import os
import random

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import sips.h as h
from sips.h import fileio as fio

from sips.macros.sports import nfl
from sips.macros.sports import nba
from sips.macros.sports import nhl


def get_dfs(folder):
    fns = fio.get_fns(folder)
    dfs = [pd.read_csv(folder + fn) for fn in fns]
    return dfs


def multivariate_data(
    dataset,
    target,
    start_index,
    end_index,
    history_size,
    target_size,
    step,
    single_step=False,
):
    '''
    create sliding window tuples for training nns on multivar timeseries
    '''
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i: i + target_size])

    return np.array(data), np.array(labels)


def remove_string_cols(df):
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


def chunk(df, cols=["game_id"], output="list"):
    # returns a python dict of dfs, splitting the df arg by unique col value
    # df type pd df, col type string
    if output == "list":
        games = [game[1] for game in df.groupby(cols)]
    elif output == "dict":
        games = {key: val for key, val in df.groupby(cols)}
    else:
        games = df.groupby(cols)
    return games


def apply_min_game_len(games, min_lines=500):
    """
    given dict of game dataframes 
    and an integer > 0 for the minimum length of a game in csv lines
    """
    print("applying minimum game len of : {}".format(min_lines))
    print("before apply: {}".format(len(games)))
    for key, value in games.copy().items():
        game_len = len(value)
        if game_len < min_lines:
            print("deleted game_id: {}".format(key))
            print("had length: {}".format(game_len))
            del games[key]
        print("after apply: {}".format(len(games)))
        return games


def label_split(df, col):
    # give column to be predicted given all others in csv
    # df is pd, col is string
    Y = df[col]
    X = df.drop(col, axis=1)
    return X, Y


def sk_scale(df, to_pd=False):
    """
    scales pandas or np data(frame) using StandardScaler 
    returns numpy or dataframe (to_pd=True)
    """
    scaler = StandardScaler()
    cols = df.columns
    if isinstance(df, pd.core.frame.DataFrame):  # pandas
        df = df.to_numpy()

    scaled = scaler.fit_transform(df)
    if to_pd:
        scaled = pd.DataFrame(scaled, columns=cols)
    return scaled


def num_flat_features(x):
    """

    """
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def apply_wins(game_df):
    """
    given a dataframe for a single game, takes the last row 
    checks if the status is 'GAME_END' 
    then adds new columns for the winner of the game based on the score
    """
    last_row = game_df.iloc[-1]
    status = last_row.status
    if status == "GAME_END":
        if last_row.a_pts > last_row.h_pts:
            a_win = True
            h_win = False
        elif last_row.a_pts < last_row.h_pts:
            a_win = False
            h_win = True
        else:
            print("game tied at end")
            a_win = False
            h_win = False
        game_df["a_win"] = a_win
        game_df["h_win"] = h_win
    else:
        print("no game end status")
    return game_df
