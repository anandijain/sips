import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

class Df(Dataset):
    def __init__(self, df, prev_n=5, next_n=1):
        df = df.astype(float)
        self.df = df.values
        self.tdf = torch.from_numpy(sk_scale(self.df))
        
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
        self.df = df.sort_values(by='cur_time')
        self.data_len = len(self.df)
        
        self.train_cols = train_cols
        self.label_cols = label_cols

        self.labels = self.df[self.label_cols]
        self.labels = self.labels.astype(float).values
        # self.labels = sk_scale(self.labels)
        self.labels = torch.tensor(self.labels)

        self.labels_shape = len(self.labels)
        if self.train_cols is None:
            self.data = self.df.drop(self.label_cols, axis=1)
        else:
            self.data = self.df[self.train_cols]
        self.data = self.data.astype(float).values
        # self.data = sk_scale(self.data)
        self.data = torch.tensor(self.data)

        self.data_shape = len(self.data[0])

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data_len