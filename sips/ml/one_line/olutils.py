import time

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sips.macros.sports import nba
from sips.h import hot

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
        self.fc2 = nn.Linear(in_dim * 2, 500)
        # self.fc3 = nn.Linear(500, 250)
        # self.fc4 = nn.Linear(250, 100)
        # self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(500, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))

        if self.classify:
            # x = self.softmax(self.fc6(x))
            x = torch.sigmoid(self.fc6(x))
        else:
            x = F.relu(self.fc6(x))

        return x


class OneLiner(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, verbose=False):
        """

        TODO: 
            compare if converting to tensor then indexing is faster than
            using dict to index by game_id then converting to tensor
        """
        self.xs, self.ys = train_dataset()
        self.length = len(self.xs)

        if verbose:
            self.__repr__()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x, y = match_rows(self.xs, self.ys, 'Game_id', idx)
        x = x.astype(np.float32)
        x = torch.tensor(x.values)
        # y = torch.tensor(y["H_win"].iloc[0], dtype=torch.float).view(-1, 1)
        y = y["H_win"].iloc[0]
        y = torch.tensor(y, dtype=torch.float).view(-1)
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


def train_dataset():
    df = train_dfs()

    wins = df[["Game_id", "H_win", "A_win"]]
    df = df.drop(["A_ML", "H_ML"], axis=1)

    df = fix_columns(df)
    df = hot_teams(df)
    # df = nums_only(df)


    return df, wins


def train_dfs(fns=FILES, how='inner'):
    df, df2 = [pd.read_csv(f) for f in fns]
    merged = df.merge(df2, on="Game_id", how=how)
    return merged


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
    cols.remove("Game_id")

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


def nums_only(df):
    df = df.select_dtypes(exclude=["object"])
    df = df.apply(pd.to_numeric)
    return df


def hot_teams(df):
    hm = hot.to_hot_map(nba.teams)

    h = hot.hot_col(df.H_team, hm)
    a = hot.hot_col(df.A_team, hm)
    a.rename(columns=lambda x: x + "_a", inplace=True)
    df = pd.concat([df, h, a], axis=1)
    df = df.drop(["A_team", "H_team"], axis=1)
    return df


def df_col_type_dict(df):
    # given a df, returns a dict where key is column name, value is dtype
    return dict(zip(list(df.columns), list(df.dtypes)))


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


def loaders(dataset, batch_size=1, train_frac=0.7, verbose=False):
    split_idx = int(len(dataset) * train_frac)
    test_len = len(dataset) - split_idx

    train_set, test_set = torch.utils.data.random_split(
        dataset, [split_idx, test_len])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    if verbose:
        data_sample(dataset)

    return train_set, test_set, train_loader, test_loader


def prep(batch_size=1, classify=True, verbose=False):
    """

    """
    dataset = OneLiner()

    x, y = dataset[0]["x"], dataset[0]["y"]

    train_set, test_set, train_loader, test_loader = loaders(
        dataset, batch_size=batch_size, verbose=verbose
    )

    in_dim = len(dataset[0]["x"])
    # if classify:
    #     out_dim = 1
    out_dim = len(dataset[0]["y"].squeeze(0))
    # out_dim = len(dataset[0]["y"].squeeze(0))

    print(f"in_dim: {in_dim}")
    print(f"out_dim: {out_dim}")

    writer = SummaryWriter(f"runs/one_liner_{time.asctime()}")
    model = Model(in_dim, out_dim).to(device)

    if classify:
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.1)
    else:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.1)

    for i, data in enumerate(train_loader, 0):

        batch_x, batch_y = data["x"].to(device), data["y"].to(device)
        optimizer.zero_grad()
        y_hat = model(batch_x)
        print(f'bx : {batch_x}')
        print(f'by : {batch_y}')
        print(f'by : {batch_y.dtype}')
        # print(f'by : {batch_y.dtype}')
        print(f'yh : {y_hat}')
        print(f'yh : {y_hat.dtype}')
        if classify:
            # loss = criterion(y_hat, torch.max(batch_y, 1)[1])
            loss = criterion(y_hat, batch_y)
        else:
            loss = criterion(y_hat, batch_y)
        break

    d = {
        "dataset": dataset,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "criterion": criterion,
        "optimizer": optimizer,
        "model": model,
        "writer": writer,
        "x": x,
        "y": y,
        "batch_x": batch_x,
        "batch_y": batch_y,
        "y_hat": y_hat,
    }
    return d
