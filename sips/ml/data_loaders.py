import os
import random

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from sips.macros import macros
from sips.ml import normdf

class Shotset(Dataset):
    def __init__(self, normed_df, feat_cols, lab_cols):        
        self.feats = feat_cols
        self.pred_cols = lab_cols

        self.df = normed_df[self.feats + self.pred_cols]
        self.length = self.df.shape[0]
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.feats].values, dtype=torch.float)
        y = torch.tensor(row[self.pred_cols].values, dtype=torch.float)
        return {'x': x, 'y': y}


class Scoreset(Dataset):
    def __init__(self, df, first_n=100, last_n=5, min_len=200):
        """
        df is already normed
        """
        games = list(df.groupby('game_id'))
        self.data = []
        for g_id, g in games:
            if g.shape[0] >= min_len:
                self.data.append(g)

        self.length = len(self.data)
        self.first_n = first_n
        self.last_n = last_n

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        g = self.data[idx]
        x_t, y_t = item_from_df(g, self.first_n, self.last_n)
        return {"x": x_t.view(-1), "y": y_t.view(-1)}


def item_from_df(df: pd.DataFrame, first_n, last_n):
    df_len = df.shape[0]
    y_idx = df_len - last_n

    early_points = [
        df.a_pts[: first_n],
        df.h_pts[: first_n],
    ]
    end_points = [df.a_pts[y_idx:], df.h_pts[y_idx:]]
    x = pd.concat(early_points)
    y = pd.concat(end_points)

    x_t = torch.tensor(x.astype(np.float32).values)
    y_t = torch.tensor(y.astype(np.float32).values)
    return x_t, y_t


def col_types(df: pd.DataFrame) -> dict:
    # given a df, returns a dict where key is column name, value is dtype
    return dict(zip(list(df.columns), list(df.dtypes)))


def train_test_ids(game_ids:list, frac=0.3):
    n = len(game_ids)

    split = int(n * frac)

    random.shuffle(game_ids)
    train_ids = game_ids[split:]
    test_ids = game_ids[:split]
    return train_ids, test_ids


def normed_scoresets(dir=macros.LINES_DIR, sport="BASK", frac=0.3, str_cols=['game_id']):
    big = scores_from_lines(dir=dir, sport=sport)
    
    games = list(big.game_id.unique())

    train_ids, test_ids = train_test_ids(games, frac=frac)

    train = big[big.game_id.isin(train_ids)]
    test = big[big.game_id.isin(test_ids)]
    
    normed_test = normdf.norm_testset(test, train, str_cols=str_cols)
    normed_train = normdf.to_normed(train, str_cols=str_cols)
    
    return normed_train.copy(), normed_test.copy()


def scores_from_lines(dir=macros.LINES_DIR, sport="BASK"):
    fns = [dir + fn for fn in os.listdir(dir)]
    dfs = [pd.read_csv(fn) for fn in fns]

    scores = pd.concat(dfs)
    # print(scores)
    scores = scores[scores.sport == sport]

    scores = scores[scores.status != "PRE_GAME"]
    scores = scores[scores.a_pts != "None"]
    scores = scores[scores.h_pts != "None"]

    scores = scores[["game_id", "h_pts", "a_pts"]]

    scores["h_pts"] = scores["h_pts"].astype(np.float32)
    scores["a_pts"] = scores["a_pts"].astype(np.float32)
    return scores

if __name__ == "__main__":
    yus = normed_scoresets()
    print(yus)
