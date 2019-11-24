import os

import pandas as pd
import numpy as np

from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from sips.h import helpers as h

# from sips.macros import macros as m


class DfWindow(Dataset):
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
        self.past = self.tdf[index - self.prev_n : index]
        # if len(self.past) < prev_n:

        if index < self.total_len - self.next_n - 1:
            self.past = self.tdf[index - self.prev_n : index]
            # self.past = torch.from_numpy(self.past)

            self.future = self.tdf[index : index + self.next_n]
            # self.future_n = torch.from_numpy(self.future_n)
            past_new = self.past.new(
                self.prev_n - self.past.size(0), *self.past.size()[1:]
            ).zero_()
            self.past = torch.cat([self.past, past_new])
            future_new = self.future.new(
                self.next_n - self.future.size(0), *self.future.size()[1:]
            ).zero_()
            self.future = torch.cat([self.future, future_new])
            return (self.past, self.future)

        else:
            return None

    def __len__(self):
        return self.total_len


class DfCols(Dataset):
    """
    each line of csv is a game,
    takes in pandas and a list of strings of which columns are labels
    """

    def __init__(
        self,
        df,
        train_cols=["quarter", "secs"],
        label_cols=["a_pts", "h_pts"],
        scale=True,
    ):
        # self.df = df.sort_values(by='cur_time')
        self.df = df
        self.data_len = len(self.df)

        self.train_cols = train_cols
        self.label_cols = label_cols

        self.labels = self.df[self.label_cols]
        # self.labels = self.labels.to_numpy(dtype=float)
        if scale:
            self.labels = h.sk_scale(self.labels)
        else:
            self.labels = self.labels.values
        self.labels = torch.tensor(self.labels)

        self.labels_shape = len(self.labels)

        if self.train_cols is None:
            self.data = self.df.drop(self.label_cols, axis=1)
        else:
            self.data = self.df[self.train_cols]

        # self.data = self.data.to_numpy(dtype=float)
        self.data = h.sk_scale(self.data)
        self.data = torch.tensor(self.data)

        self.data_shape = len(self.data[0])

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data_len


class LineGen(Dataset):
    def __init__(self, game_df):
        self.df = game_df
        self.df = self.df[
            (self.df.a_odds_ml != np.int64(0)) & (self.df.h_odds_ml != np.int64(0))
        ]
        self.data_len = len(self.df)

        min_max_scaler = preprocessing.MinMaxScaler()

        self.pregame = self.df[["a_odds_ps", "h_odds_ps"]].iloc[0].values
        self.train_cols = [
            "last_mod_to_start",
            "a_pts",
            "h_pts",
            "quarter",
            "status",
            "secs",
            "num_markets",
        ]
        self.label_cols = ["a_odds_ml", "h_odds_ml"]
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
    def __init__(
        self,
        window=1,
        fn="./data/static/lets_try5.csv",
        predict_columns=["pass_rating", "pass_yds", "rush_yds", "rec_yds"],
        team_columns=None,
    ):
        self.projections_frame = pd.read_csv(fn)
        self.transform(self.projections_frame)
        self.team_columns = team_columns
        self.predict_columns = predict_columns
        self.all_cols = self.predict_columns + self.team_columns
        self.window = window

    def transform(self, df):
        self.projections_frame = self.projections_frame.drop_duplicates()
        dfs = self.projections_frame.groupby("playerid")
        bigcsv = []
        for i in dfs:
            df = i[1]
            length = len(df)
            df = df.sort_values(by=["age"])
            if df.pass_yds.astype(bool).sum(axis=0) > 0.4 * length:
                bigcsv.append(df)
        self.projections_frame = pd.concat(bigcsv)

    def __len__(self):
        return len(self.projections_frame)

    def __getitem__(self, index):
        upper = index + self.window
        past = torch.tensor(self.projections_frame[index:upper][self.all_cols].values)
        past = torch.tensor(past.reshape(-1, 1))
        team_data = torch.tensor(self.projections_frame.iloc[upper][self.team_columns])
        # past = past.t()
        y = torch.tensor(
            self.projections_frame.iloc[index + self.window][self.predict_columns]
        )
        past = past.float()
        team_data = team_data.float()
        x = torch.cat((past.flatten(), team_data.flatten())).view(1, 1, -1)

        tup = (x, y)
        return tup


class LinesLoader(Dataset):
    def __init__(self, dir="../data/", fn="nfl_history_no_strings.csv"):
        df = pd.read_csv(dir + fn)
        df = df.fillna(0)
        print(df)
        # print(df)
        # for i, column in enumerate(df.columns):
        #     print(f'{i}: {column}')

        # for i, dtype in enumerate(df.dtypes):
        #     print(f'{i}: {dtype}')

        self.df = pd.get_dummies(df, columns=["a_team", "h_team", "Venue"])
        self.df = h.remove_string_cols(self.df)
        print(self.df)

        self.target_columns = ["a_win", "h_win"]
        self.X = self.df.drop(self.target_columns, axis=1)
        self.y = self.df[self.target_columns]
        self.length = len(self.df)
        self.dtypes = self.X.dtypes

    def __getitem__(self, index):
        X = torch.tensor(self.X.iloc[index].values, dtype=torch.float32)
        y = torch.tensor(self.y.iloc[index].values)
        return X, y

    def __len__(self):
        return self.length


class WinSet(Dataset):
    def __init__(
        self,
        predict_column=["h_win"],
        train_columns=[
            "gen_avg_allowed",
            "gen_avg_pass_comp_pct",
            "gen_avg_pass_yards",
            "gen_avg_rush_yards",
            "gen_avg_rush_yards_per_attempt",
            "gen_avg_score",
            "gen_avg_total_yards",
        ],
    ):
        self.predict_col = predict_column
        self.train_cols = train_columns

        df = pd.read_csv("./data/static/big_daddy2.csv")
        labels = df["h_win"].copy()
        df = df[df.Gen_Games > 4]
        x = df[self.train_cols].values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled, columns=self.train_cols)
        self.projections_frame = pd.concat((df, labels), axis=1).fillna(0)

    def __len__(self):
        return len(self.projections_frame)

    def __getitem__(self, index):
        x = torch.tensor(
            self.projections_frame.iloc[index][self.train_cols], dtype=torch.float
        )
        y = self.projections_frame.iloc[index][self.predict_col].values
        if y == 1:
            y = torch.tensor([0.0, 1.0], dtype=torch.float)
        elif y == 0:
            y = torch.tensor([1.0, 0.0], dtype=torch.float)
        else:
            y = torch.tensor([0.0, 0.0], dtype=torch.float)

        tup = (x, y)
        return tup


class VAELoader(Dataset):
    def __init__(self):
        self.df = pd.read_csv("./data/nba2.csv")
        self.cols = [
            "game_id",
            "cur_time",
            "quarter",
            "secs",
            "a_pts",
            "h_pts",
            "status",
            "a_win",
            "h_win",
            "last_mod_to_start",
            "last_mod_lines",
            "num_markets",
            "a_odds_ml",
            "h_odds_ml",
            "a_odds_ps",
            "h_odds_ps",
            "a_hcap_ps",
            "h_hcap_ps",
            "game_start_time",
        ]

        self.df_parsed = self.df[self.cols]
        group = self.df_parsed.groupby(["game_id", "quarter"])

        # might be a torch fxn to find max seq len
        max = 0
        grouped = []
        for elt in group:
            cur_len = len(elt)

            if cur_len > max:
                max = cur_len
            grouped.append(torch.tensor(elt[1].values, dtype=torch.double))

        self.grouped = grouped

        pad = 0
        if max % 2 != 0:
            pad = 1

        self.padded = nn.utils.rnn.pad_sequence(grouped, padding_value=pad)
        print(self.padded.shape)
        item = self.padded[0]

        self.length = len(item.flatten())

    def __getitem__(self, index):
        return self.padded[index].flatten()

    def __len__(self):
        return self.length


class FileLoader:
    def __init__(self, directory):
        self.files = os.listdir(directory)
        self.length = len(self.files)
        self.dir = directory
        self.file = self.files[0]

    def __getitem__(self, index):
        self.file = self.files[index]
        df = pd.read_csv(self.dir + self.files[index])
        return df.iloc[:, 1:5].values

        # x = df.values #returns a numpy array
        # min_max_scaler = preprocessing.MinMaxScaler()
        # x_scaled = min_max_scaler.fit_transform(x)
        # data = x_scaled

    def __len__(self):
        return self.length


class LSTMLoader(Dataset):
    def __init__(self, data, window_len=1, predict_window=1):
        self.samples = []
        self.length = len(data)
        self.window_len = window_len
        self.predict_window = predict_window
        self.data = data
        self.get_data()

    def get_data(self):
        for i in range(0, self.length - self.predict_window):
            upper_idx = i + self.window_len
            x = torch.tensor(self.data[i:upper_idx, :]).view(1, 1, -1).float()
            y = (
                torch.tensor(self.data[upper_idx : upper_idx + self.predict_window, :])
                .view(1, 1, -1)
                .float()
            )
            self.samples.append((x, y))

    def __len__(self):
        return self.length - self.predict_window  # (self.window_len + 1)

    def __getitem__(self, index):
        return self.samples[index]
