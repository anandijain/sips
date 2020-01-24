import time

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sips.macros.sports import nba
from sips.h import hot
from sips.ml import normdf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FILES = [
    "/home/sippycups/absa/sips/data/nba/nba_history.csv",
    "/home/sippycups/absa/sips/data/nba/nba_history_with_stats.csv",
]


class Model(nn.Module):
    def __init__(self, in_dim, out_dim, classify=True):
        super(Model, self).__init__()
        self.classify = classify
        self.fc1 = nn.Linear(in_dim, in_dim * 2)
        self.fc2 = nn.Linear(in_dim * 2, 250)
        # self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 100)
        # self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(f'X ONE {x}')
        x = torch.relu(self.fc1(x))
        # print(x)
        x = self.fc2(x)
        # print(x)
        # x = self.fc3(x)
        # print(x)
        x = self.fc4(x)
        # print(x)
        # x = self.fc5(x)

        if self.classify:
            x = self.softmax(self.fc6(x))
            # x = torch.sigmoid(self.fc6(x))
        else:
            x = F.relu(self.fc6(x))

        return x


class OneLiner(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, xs, ys, norm=True, verbose=False):
        """

        TODO: 
            compare if converting to tensor then indexing is faster than
            using dict to index by game_id then converting to tensor
        """

        self.xs = xs  # .drop(['H_win', 'A_win'], axis=1)
        self.ys = ys
        # print(ys)
        # print(f'ys.dtype: {type(ys)}')
        self.length = len(self.xs)

        if verbose:
            self.__repr__()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x, y = match_rows(self.xs, self.ys, 'Game_id', idx)

        x = x.astype(np.float32)
        x = torch.tensor(x.values)

        y = y[["H_win", "A_win"]].iloc[0]
        y = torch.tensor(y.values, dtype=torch.float).view(-1)
        return {"x": x, "y": y}

    def __repr__(self):
        return f'xs: {self.xs} \
                ys_shape: {self.ys.shape} \
                len: {self.length}'


def match_rows(df, df2, col, idx):
    x = df.iloc[idx]
    match_val = x[col]
    x = x.drop(col)
    y = df2[df2[col] == match_val]
    return x, y




def train_test_dfs(frac=0.7, verbose=True):
    df = train_dfs()
    length = df.shape[0]

    df = sklearn.utils.shuffle(df)

    split = int(length * frac)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    if verbose:
        print(f'df: {df}')
        print(f'train {train.head()} {train.shape}')
        print(f'test {test.head()} {test.shape}')
    return train, test


def train_test_sets(train, test, frac=0.3):

    train_wins = train[["Game_id", "H_win", "A_win"]].copy()
    test_wins = test[["Game_id", "H_win", "A_win"]].copy()

    train = train.drop(["A_win", "H_win"], axis=1)
    test = test.drop(["A_win", "H_win"], axis=1)

    test = normdf.norm_testset(test, train)
    train = normdf.to_normed(train)

    # train_x, test_x, train_y, test_y = train_test_split(df, wins, test_size=frac)

    train = hot_teams(train)
    test = hot_teams(test)

    train_set = OneLiner(train, train_wins)
    test_set = OneLiner(test, test_wins)

    return train_set, test_set


def train_dfs(fns=FILES, how='inner') -> pd.DataFrame:
    df, df2 = [pd.read_csv(f) for f in fns]
    merged = df.merge(df2, on="Game_id", how=how)
    merged = merged.drop(["A_ML", "H_ML"], axis=1)
    merged = merged.dropna()
    merged = fix_columns(merged)

    return merged


def hot_teams(df, cols=['H_team', 'A_team']):
    hm = hot.to_hot_map(nba.teams)
    h = hot.hot_col(df[cols[0]], hm)
    a = hot.hot_col(df[cols[1]], hm)

    a.rename(columns=lambda x: x + "_a", inplace=True)
    df = pd.concat([df, h, a], axis=1)
    df = df.drop(["A_team", "H_team"], axis=1)
    return df


def data_sample(dataset):
    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample["x"], sample["y"])
        if i == 0:
            x_shape = sample["x"].shape
            y_shape = sample["y"].shape
            print(f"x_shape: {x_shape}")
            print(f"y_shape: {y_shape}")
            break


def fix_columns(df):
    df = df.rename(
        columns={
            "A_team_x": "A_team",
            "H_team_x": "H_team",
            "Date_x": "Date",
            "Season_x": "Season",
        }
    )
    cols = list(df.columns)
    for x in ["Game_id", 'H_win', 'A_win']:
        cols.remove(x)

    for col in nba.POST_GAME:
        if col in cols:
            df = df.drop(col, axis=1)

    df = df.rename(
        columns={
            "A_team_y": "A_team",
            "H_team_y": "H_team",
            "Date_y": "Date",
            "Season_y": "Season",
        }
    )
    return df


def prep(batch_size=1, classify=True, verbose=False):
    """

    """
    train_df, test_df = train_test_dfs()
    dataset, test_set = train_test_sets(train_df, test_df)
    data_sample(dataset)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=4
    )

    x, y = dataset[0]["x"], dataset[0]["y"]

    in_dim = len(dataset[0]["x"])
    if classify:
        out_dim = 2

    # out_dim = len(dataset[0]["y"].squeeze(0))

    print(f"in_dim: {in_dim}")
    print(f"out_dim: {out_dim}")

    writer = SummaryWriter(f"runs/one_liner_{time.asctime()}")
    model = Model(in_dim, out_dim).to(device)

    if classify:
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters())
    else:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for i, data in enumerate(train_loader, 0):

        batch_x, batch_y = data["x"].to(device), data["y"].to(device)
        optimizer.zero_grad()
        y_hat = model(batch_x)
        if classify:
            loss = criterion(y_hat, torch.max(batch_y, 1)[1])
            # loss = criterion(y_hat, batch_y)
        else:
            loss = criterion(y_hat, batch_y)
        break

    d = {
        "dataset": dataset,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "train_df": train_df,
        "test_df": test_df,
        "criterion": criterion,
        "optimizer": optimizer,
        "model": model,
        "writer": writer,
        "x": x,
        "y": y,
        "batch_x": batch_x,
        "batch_y": batch_y,
        "y_hat": y_hat,
        'classify': classify
    }
    return d


if __name__ == "__main__":
    train, test = train_test_dfs()
    tr, te = train_test_sets(train, test)
    print(f'tr: {tr}')
    print(f'te: {te}')
