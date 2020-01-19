import time

import pandas as pd
import numpy as np

import sklearn

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sips.macros.sports import nba
from sips.h import hot


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

PATH = "./one_liner.pth"

FILES = [
    "/home/sippycups/absa/sips/data/nba/nba_history.csv",
    "/home/sippycups/absa/sips/data/nba/nba_history_with_stats.csv",
]


COLS = [
    "Home_team_gen_avg_3_point_attempt_rate",
    "Home_team_gen_avg_3_pointers_attempted",
    "Home_team_gen_avg_3_pointers_made",
    "Home_team_gen_avg_assist_percentage",
    "Home_team_gen_avg_assists",
    "Home_team_gen_avg_block_percentage",
    "Home_team_gen_avg_blocks",
    "Home_team_gen_avg_defensive_rating",
    "Away_team_gen_avg_3_point_attempt_rate",
    "Away_team_gen_avg_3_pointers_attempted",
    "Away_team_gen_avg_3_pointers_made",
    "Away_team_gen_avg_assist_percentage",
    "Away_team_gen_avg_assists",
    "Away_team_gen_avg_block_percentage",
    "Away_team_gen_avg_blocks",
    "Away_team_gen_avg_defensive_rating",
    "Away_team_gen_avg_defensive_rebound_percentage",
    "Away_team_gen_avg_defensive_rebounds",
]


class Model(nn.Module):
    def __init__(self, in_dim, out_dim, classify=True):
        super(Model, self).__init__()
        self.classify = classify
        self.fc1 = nn.Linear(in_dim, in_dim - 1)
        self.fc2 = nn.Linear(in_dim - 1, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))

        if self.classify:
            x = self.softmax(self.fc6(x))
        else:
            x = F.relu(self.fc6(x))

        return x


class OneLiner(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, norm=True, shuffle=True, verbose=True):
        """

        TODO: 
            compare if converting to tensor then indexing is faster than
            using dict to index by game_id then converting to tensor
        """
        self.xs, self.ys = clean(hot_data=True)
        self.length = len(self.xs)

        if verbose:
            print(self.xs)
            print(self.ys.shape)
            print(self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.xs.iloc[idx]
        game_id = x.Game_id

        x = x.drop("Game_id").astype(np.float32)
        x = torch.tensor(x.values, dtype=torch.float)

        y = self.ys[self.ys["Game_id"] == game_id]
        y = torch.tensor(y[["H_win", "A_win"]].iloc[0].values)
        return {"x": x, "y": y}


def get_data(fn):
    X = pd.read_csv(fn)

    X = X.select_dtypes(exclude=["object"])
    X = X.apply(pd.to_numeric, errors="coerce")

    hot.hot(X, nba.teams)

    return X


def clean(hot_data=True, nums_only=False, how="inner", coerce=True):
    df, df2 = [pd.read_csv(f) for f in FILES]
    merged = df.merge(df2, on="Game_id", how=how)

    wins = df[["Game_id", "H_win", "A_win"]]

    merged = merged.drop(["A_ML", "H_ML"], axis=1)

    merged = merged.rename(
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
            merged = merged.drop(col, axis=1)
    merged = merged.rename(
        columns={
            "A_team_y": "A_team",
            "H_team_y": "H_team",
            "Date_y": "Date",
            "Season_y": "Season",
        }
    )

    if nums_only:
        merged = merged.select_dtypes(exclude=["object"])
        if coerce:
            merged = merged.apply(pd.to_numeric, errors="coerce")
        else:
            merged = merged.apply(pd.to_numeric)
        types = df_col_type_dict(merged)

    print(merged)
    if hot_data:
        hm = hot.to_hot_map(nba.teams)

        h = hot.hot_col(merged.H_team, hm)
        a = hot.hot_col(merged.A_team, hm)
        a.rename(columns=lambda x: x + "_a", inplace=True)
        merged = pd.concat([merged, h, a], axis=1)
        merged = merged.drop(["A_team", "H_team"], axis=1)

        # TODO:
        merged = merged.dropna()
    return merged, wins


def df_col_type_dict(df):
    # given a df, returns a dict where key is column name, value is dtype
    return dict(zip(list(df.columns), list(df.dtypes)))


def loaders(dataset, batch_size=1, verbose=False):
    split_idx = len(dataset) // 2
    test_len = len(dataset) - split_idx

    train_set, test_set = torch.utils.data.random_split(
        dataset, [split_idx, test_len])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    if verbose:
        for i in range(len(dataset)):
            sample = dataset[i]

            print(i, sample["x"], sample["y"])
            if i == 0:
                x_shape = sample["x"].shape
                y_shape = sample["y"].shape
                print(f"x_shape: {x_shape}")
                print(f"y_shape: {y_shape}")
                break
    return train_set, test_set, train_loader, test_loader


def prep(batch_size=5, classify=True, verbose=False):
    """

    """
    dataset = OneLiner()
    x, y = dataset[0]["x"], dataset[0]["y"]
    train_set, test_set, train_loader, test_loader = loaders(
        dataset, batch_size=batch_size, verbose=verbose
    )

    in_dim = len(dataset[0]["x"])
    out_dim = len(dataset[0]["y"].squeeze(0))

    print(f"in_dim: {in_dim}")
    print(f"out_dim: {out_dim}")

    writer = SummaryWriter(f"runs/one_liner_{time.asctime()}")
    model = Model(in_dim, out_dim)

    if classify:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for i, data in enumerate(train_loader, 0):

        batch_x, batch_y = data["x"].to(device), data["y"].to(device)
        optimizer.zero_grad()
        y_hat = model(batch_x)
        break
        if classify:
            loss = criterion(y_hat, torch.max(batch_y, 1)[1])
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
        # "loss": loss,
    }
    return d


def train():
    BATCH_SIZE = 1
    CLASSIFY = True
    d = prep(classify=CLASSIFY, batch_size=BATCH_SIZE)
    dataset = d["dataset"]
    train_loader = d["train_loader"]
    test_loader = d["test_loader"]
    model = d["model"]
    writer = d["writer"]
    criterion = d["criterion"]
    optimizer = d["optimizer"]

    running_loss = 0
    for epoch in range(2):
        running_loss = 0.0
        print("training")
        model.train()
        for i, data in enumerate(train_loader, 0):
            x, y = data["x"].to(device), data["y"].to(device)

            optimizer.zero_grad()

            y_hat = model(x.reshape(BATCH_SIZE, -1).float())

            loss = criterion(y_hat, torch.max(y, 1)[1])
            loss.backward()
            optimizer.step()

            writer.add_scalar("train_loss", loss, i +
                              epoch * len(train_loader))
            writer.add_scalar

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
                print(f"y: {y}, y_hat: {y_hat}")

                running_loss = 0.0

        print("testing")
        model.eval()

        running_loss = 0.0
        for j, test_data, in enumerate(test_loader, 0):
            test_x, test_y = test_data["x"].to(
                device), test_data["y"].to(device)

            optimizer.zero_grad()

            test_y_hat = model(test_x.reshape(1, -1).float())
            test_loss = criterion(test_y_hat, torch.max(test_y, 1)[1])

            writer.add_scalar("test_loss", test_loss, j +
                              epoch * len(test_loader))

            running_loss += test_loss.item()
            if j % 2000 == 1999:
                print(f"[{epoch + 1}, {j + 1}] loss: {running_loss / 2000}")
                print(f"test_y: {test_y}, test_y_hat: {test_y_hat}")
                running_loss = 0.0

        torch.save(model.state_dict(), PATH)

    print("Finished Training")


if __name__ == "__main__":

    train()
