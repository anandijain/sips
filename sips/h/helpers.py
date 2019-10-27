import os
import random
from pathlib import Path

import pandas as pd
import numpy as np

import torch
from sklearn.preprocessing import StandardScaler

import sips.h as h
# todo trash this code 

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

def full_fn(fn):
    str_path = './sips/gym_sip/data/static/' + fn + '.csv'
    # path = Path(str_path)
    # return os.path.join(sips.__path__, path)
    return str_path


def df_cols():
    df = get_df()
    try:
        loader = h.DfCols(df)
    except Exception:
        loader = None
        pass
    return df, loader


def df_combine(fn='come2017season', fn2='come2015season'):
    complete_fns = [full_fn(full_name) for full_name in [fn, fn2]]
    df1 = pd.read_csv(complete_fns[0])
    df2 = pd.read_csv(complete_fns[1])

    df1 = df1.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    df2 = df2.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    df3 = pd.concat([df2, df1])
    df3 = pd.get_dummies(data=df3)
    return df3


def get_games(fn='mlb', output='list'):
    # takes in fn and returns python dict of pd dfs
    # TODO allow get_games to take in either a df or a fn
    out_type = output
    df = get_df(fn)
    games = chunk(df, output=out_type)
    games = remove_missed_wins(games)
    return games


def get_df(fn='mlb', dummies=['league', 'a_team', 'h_team'], one_hot=False, scale_times=False):
    # complete_fn = full_fn(fn)
    df = csv(fn)

    if one_hot:
        df = pd.get_dummies(data=raw, columns=dummies, sparse=False)

    df = drop_null_times(df)
    if scale_times:
        df = scaled_times(df)

    df = df.drop(df.select_dtypes(object), axis=1)
    return df


def chunk(df, cols=['game_id'], output='list'):
    # returns a python dict of pandas dfs, splitting the df arg by unique col value
    # df type pd df, col type string
    if output == 'list':
        games = [game[1] for game in df.groupby(cols)]
    elif output == 'dict':
        games = {key: val for key, val in df.groupby(cols)}
    else:
        games = df.groupby(cols)
    return games


def csv(fn='mlb'):
    # takes in file name string, returns pandas dataframe
    print(fn)
    df = pd.read_csv(full_fn(fn), error_bad_lines=False)
    df = df.dropna()
    # df = df.drop('sport', axis=1)
    return df


def one_hots(df, cols):
    # df is pandas df, cols is string list
    one_hot_df = pd.get_dummies(data=df, columns=cols, sparse=False)
    return one_hot_df


def remove_missed_wins(games):
    # takes in a dictionary or list of df games
    games_len = len(games)
    if isinstance(games, dict):
        for g_id in list(games.keys()):
            if len(games[g_id]['a_win'].unique()) + len(games[g_id]['h_win'].unique()) != 3:
                del games[g_id]

    elif isinstance(games, list):
        for i, elt in enumerate(games):
            if len(elt['a_win'].unique()) + len(elt['h_win'].unique()) != 3:
                del games[i]
    else:
        raise TypeError('games argument must be dict or list')

    if games_len != len(games):
        print('before: {}'.format(games_len))
        print('after: {}'.format(len(games)))

    return games


def drop_null_times(df, columns=['lms_date', 'lms_time']):
    # given pandas df and list of strings for columns. convert '0' values to np.datetime64
    init_len = len(df)
    for col in columns:
        df[col] = df[col].replace('0', np.nan)

        df = df.dropna()

        after_len = len(df)
        delta = init_len - after_len

        print('len(df) before: {}, after length: {}, delta: {}'.format(init_len, after_len, delta))
    return df


def dates(df):
    # convert ['lms_date', 'lms_time'] into datetimes
    # df['datetime'] = df['lms_date'] + ' ' + df['lms_time']
    # df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = pd.to_datetime(df['date'])

    dt = df['date'].dt
    df['time_vals'] = pd.to_numeric(df['date'])
    date_categories = [dt.year, dt.month, dt.week, dt.day,
                       dt.hour, dt.minute, dt.second, dt.dayofweek, dt.dayofyear]
    col_names = ['year', 'month', 'week', 'day', 'hour', 'minute', 'second', 'dayofweek', 'dayofyear']

    for i in range(len(date_categories) - 1):
        df[col_names[i]] = date_categories[i]

        # print(df.columns)
    # df = df.drop(['lms_date', 'lms_time', 'datetime'], axis=1)
    df = df.drop(['date'], axis=1)

    # print('df after h.dates: {}'.format(df))
    return df


def scaled_times(df):
    df['date'] = df['lms_date'] + ' ' + df['lms_time']
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = pd.to_numeric(df['date'])
    df['date'] = (df['date'] - df['date'].mean()) / (df['date'].max() - df['date'].min())
    df = df.drop(['lms_date', 'lms_time'], axis=1)

    return df


def split_game_in_half(game):
    game_len = len(game)
    first_half = game[:game_len//2, ]
    second_half = game[game_len//2:, ]
    return first_half, second_half


def apply_min_game_len(games, min_lines=500):
    # given dict of game dataframes and an integer > 0 for the minimum length of a game in csv lines
    print('applying minimum game len of : {}'.format(min_lines))
    print('before apply: {}'.format(len(games)))
    for key, value in games.copy().items():
        game_len = len(value)
        if game_len < min_lines:
            print('deleted game_id: {}'.format(key))
            print('had length: {}'.format(game_len))
            del games[key]
        print('after apply: {}'.format(len(games)))
        return games


def random_game(games):
    game_id, game = random.choice(list(games.items()))
    print('game_id: {}'.format(game_id))
    return game


def label_split(df, col):
    # give column to be predicted given all others in csv
    # df is pd, col is string
    Y = df[col]
    X = df.drop(col, axis=1)
    return X, Y


def train_test(df, train_pct=0.5):
    # takes in pandas or np array and returns corresponding type
    if isinstance(df, pd.core.frame.DataFrame):  # pandas
        train = df.sample(frac=train_pct, random_state=None)
        test = df.drop(train.index)

        return train, test
    elif isinstance(df, np.ndarray): # np
        length = len(df)
        split_index = round(length * train_pct)

        train = df[:split_index, :]
        test = df[split_index:, :]

        return train, test
    else:
        raise TypeError("please provide a numpy array or a pandas dataframe")


def sk_scale(df):
    scaler = StandardScaler()

    if isinstance(df, pd.core.frame.DataFrame):  # pandas
        df = df.to_numpy()

    scaled = scaler.fit_transform(df)
    return scaled


def select_dtypes(df, dtypes=['number']):
    return df.select_dtypes(include=dtypes)


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def games_to_padded_tensors(games, length=1200):
    # takes in games dict and returns dict of padded length tensors
    copy = games.copy()
    for game in copy:
        game_df = games[game]
        game_np = game_df.values
        games[game] = pad_tensor(game_np)
    return games


def get_t_games(df):
    games = chunk(df)
    games = remove_missed_wins(games)
    t_games = games_to_padded_tensors(games)
    return t_games


def pad_tensor(game, length=1200):
    tensor = torch.from_numpy(game)
    return torch.cat([tensor, tensor.new(length - tensor.size(0), * tensor.size()[1:]).zero_()])


def onehot_teams(a_team, h_team):
    # take in 2 teams playing and onehot to alphabetical one_hots, returns series
    one_hotted = []

    for cur_team in [a_team, h_team]:
        for index, elt in enumerate(h.macros.az_teams):
            if cur_team == elt:
                a = pd.Series([0 for i in range(index)])
                b = pd.Series([1])
                c = pd.Series([0 for i in range(30-index-1)])

    half = pd.concat([a, b, c])
    half = half.reset_index(drop=True)

    one_hotted.append(half)
    return one_hotted
