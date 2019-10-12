import pandas as pd
import numpy as np

from sklearn import preprocessing

import torch
from torch.utils.data import Dataset

from sips.h.helpers import *

class Df(Dataset):
    def __init__(self, df, prev_n=5, next_n=1):
        df = df.astype(float)
        self.df = df.values()
        # self.tdf = torch.from_numpy(sk_scale(self.df))
        self.tdf = torch.to_tensor(self.df)

        self.total_len = len(self.tdf)
        self.num_cols = len(self.df[0])

        self.prev_n = prev_n
        self.next_n = next_n

        self.past = None
        self.future = None

        item = self.__getitem__(500)

        self.shape = item[0].shape
        self.out_shape = item[1].shape

    def __getitem__(self, index):
        self.past = self.tdf[index - self.prev_n:index]
        # if len(self.past) < prev_n:

        if index < self.total_len - self.next_n - 1:
            self.past = self.tdf[index - self.prev_n : index]
            # self.past = torch.from_numpy(self.past)

            self.future = self.tdf[index: index + self.next_n]
            # self.future_n = torch.from_numpy(self.future_n)

            self.past = torch.cat([self.past, self.past.new(self.prev_n - self.past.size(0), * self.past.size()[1:]).zero_()])
            self.future = torch.cat([self.future, self.future.new(self.next_n - self.future.size(0), * self.future.size()[1:]).zero_()])
            return (self.past, self.future)

        else:
            return None

    def __len__(self):
        return self.total_len


class DfGame(Dataset):
    # given dictionary of padded games
    def __init__(self, games):
        self.games_dict = games
        self.games = list(self.games_dict.values())
        self.ids = list(self.games_dict.keys())
        self.game_id = self.ids[0]
        self.game = self.games[0]
        self.data_len = len(self.games)  # not padding aware because shape is constant
        self.game_len = len(self.games_dict[self.game_id])  # assuming all games same length

    def __getitem__(self, index):
        self.game = self.games[index]
        first_half, second_half = split_game_in_half(self.game)
        return self.game

    def __len__(self):
        return self.data_len


class DfPastGames(Dataset):
    # each line of csv is a game, takes in pandas and a list of strings of which columns are labels
    def __init__(self, df, train_columns=['a_pts', 'h_pts']):
        self.df = df
        self.train_columns = train_columns
        self.data_len = len(self.df)

        self.labels = self.df[self.train_columns].values
        # self.labels= sk_scale(self.labels)
        self.labels = torch.tensor(self.labels)
        self.labels_shape = len(self.labels)

        self.data = self.df.drop(self.train_columns, axis=1).values
        # self.data = sk_scale(self.data)
        print(self.data)
        self.data = torch.tensor(self.data)
        self.data_shape = len(self.data[0])

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data_len


class DfCols(Dataset):
    # each line of csv is a game, takes in pandas and a list of strings of which columns are labels
    def __init__(self, df, train_cols=['quarter', 'secs'], label_cols=['a_pts', 'h_pts']):
        # self.df = df.sort_values(by='cur_time')
        self.df = df
        self.data_len = len(self.df)

        self.train_cols = train_cols
        self.label_cols = label_cols

        self.labels = self.df[self.label_cols]
        # self.labels = self.labels.to_numpy(dtype=float)
        self.labels = sk_scale(self.labels)
        self.labels = torch.tensor(self.labels)

        self.labels_shape = len(self.labels)

        if self.train_cols is None:
            self.data = self.df.drop(self.label_cols, axis=1)
        else:
            self.data = self.df[self.train_cols]

        # self.data = self.data.to_numpy(dtype=float)
        self.data = sk_scale(self.data)
        self.data = torch.tensor(self.data)

        self.data_shape = len(self.data[0])

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data_len


class LineGen(Dataset):
    def __init__(self, game_df):
        self.df = game_df
        self.df = self.df[(self.df.a_odds_ml != np.int64(0)) & (self.df.h_odds_ml != np.int64(0))]
        self.data_len = len(self.df)

        min_max_scaler = preprocessing.MinMaxScaler()

        self.pregame = self.df[['a_odds_ps', 'h_odds_ps']].iloc[0].values
        self.train_cols = ['last_mod_to_start', 'a_pts', 'h_pts', 'quarter', 'status', 'secs', 'num_markets']
        self.label_cols = ['a_odds_ml', 'h_odds_ml']
        self.X = self.df[self.train_cols].values
        self.Y = self.df[self.label_cols].values

        # self.X = min_max_scaler.fit_transform(self.X)

        self.tups = []
        self.gen_tups()

    def __getitem__(self, index):
        return self.tups[index]

    def __len__(self):
        return self.data_len

    def gen_tups(self):
        for i in range(self.data_len):
            pg = torch.tensor(self.pregame, dtype=torch.float)
            x = torch.cat((torch.tensor(self.X[i], dtype=torch.float), pg))
            y = torch.tensor(self.Y[i], dtype=torch.float)
            self.tups.append((x, y))

class PlayerDataset(Dataset):
    def __init__(self, window=1, fn='./data/static/lets_try5.csv', predict_columns =['pass_rating', 'pass_yds', 'rush_yds', 'rec_yds'], team_columns=None):
        self.projections_frame = pd.read_csv(fn)
        self.transform(self.projections_frame)
        self.team_columns = team_columns
        self.predict_columns = predict_columns
        self.window = window
    def transform(self, df):
        self.projections_frame = self.projections_frame.drop_duplicates()
        dfs = self.projections_frame.groupby('playerid')
        bigcsv = []
        for i in dfs:
            df = i[1]
            length = len(df)
            df = df.sort_values(by=['age'])
            if df.pass_yds.astype(bool).sum(axis=0) > .4*length:
                    bigcsv.append(df)
        self.projections_frame = pd.concat(bigcsv)

    def __len__(self):
        return len(self.projections_frame)

    def __getitem__(self, index):
        past = torch.tensor(np.array(self.projections_frame[index:index+self.window][self.predict_columns + self.team_columns]))
        past = torch.tensor(past.reshape(-1, 1))
        team_data = torch.tensor(self.projections_frame.iloc[index+self.window][self.team_columns])
        # past = past.t()
        y = torch.tensor(self.projections_frame.iloc[index+self.window][self.predict_columns])
        past = past.float()
        team_data = team_data.float()
        x = torch.cat((past.flatten(), team_data.flatten())).view(1, 1, -1)

        tup = (x, y)
        return tup
