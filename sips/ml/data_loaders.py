import os
import random

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from sips.macros import macros
from sips.ml import normdf


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


def normed_scoresets(dir=macros.LINES_DIR, sport="BASK", frac=0.3):
    fns = [dir + fn for fn in os.listdir(dir)]
    # str_cols = ["game_id", "h_team", "h_team"]
    str_cols = ["game_id"]
    dfs = [pd.read_csv(fn) for fn in fns]

    big = pd.concat(dfs)
    print(big)
    big = big[big.sport == sport]

    big = big[big.status != "PRE_GAME"]
    big = big[big.a_pts != "None"]
    big = big[big.h_pts != "None"]

    # big = big[["game_id", "h_team", "a_team", "h_pts", "a_pts"]]
    big = big[["game_id", "h_pts", "a_pts"]]

    big["h_pts"] = big["h_pts"].astype(np.float32)
    big["a_pts"] = big["a_pts"].astype(np.float32)
    print(col_types(big))
    print(big)
    games = list(big.game_id.unique())
    n = len(games)

    split = int(n * frac)

    random.shuffle(games)
    train_ids = games[split:]
    test_ids = games[:split]

    train = big[big.game_id.isin(train_ids)]
    test = big[big.game_id.isin(test_ids)]
    print(col_types(train))
    print(col_types(test))
    normed_test = normdf.norm_testset(test, train, str_cols=str_cols)
    normed_train = normdf.to_normed(train, str_cols=str_cols)
    return normed_train.copy(), normed_test.copy()


if __name__ == "__main__":
    normed_scoresets()
