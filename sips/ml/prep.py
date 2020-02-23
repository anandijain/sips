import time
import random

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset


from sips.h import calc
from sips.h import helpers
from sips.ml import normdf
from sips.ml import models
from sips.ml.one_line import olutils


def fn_to_tr_te(fn, by: str, xcols: list = None, ycols: list = None, norm_y=False):
    df = pd.read_csv(fn)
    df = df.dropna()
    df = df[xcols]

    df = df.sample(frac=1).reset_index(drop=True)
    if norm_y:
        tr, te = normdf.splitnorm(df, y_cols=ycols, by=by)
    else:
        tr, te = normdf.split_norm(df, y_cols=ycols, by=by)
    
    tr = tr.reset_index(drop=True)
    te = te.reset_index(drop=True)
    return tr, te


def prep_sportset(df, to_dummies, sport: str, train_frac=0.7):
    sportdf = df[df.sport == sport]
    sportdf = sportdf.drop("sport", axis=1)

    tr_df, te_df = helpers.split_by(sportdf, train_frac=train_frac)

    test_normed = normdf.norm_testset(te_df, tr_df, str_cols=to_dummies)
    train_normed = normdf.to_normed(tr_df, str_cols=to_dummies)

    return train_normed, test_normed


def clean(df, to_drop=["a_ou", "h_ou"]):
    df = df.drop(to_drop, axis=1)
    df = df[df != "None"]
    df = df.dropna()
    df = df.replace("EVEN", 100)
    df.reset_index(inplace=True, drop=True)
    return df


def main(train_frac=0.7):
    to_nums = [
        "secs",
        "live",
        "a_pts",
        "h_pts",
        "a_ps",
        "h_ps",
        "a_hcap",
        "h_hcap",
        "a_ml",
        "h_ml",
        "a_tot",
        "h_tot",
        "a_hcap_tot",
        "h_hcap_tot",
    ]
    ml_cols = ["a_hcap", "h_hcap", "a_ml", "h_ml", "a_hcap_tot", "h_hcap_tot"]
    to_dummies = [
        "a_team",
        "h_team",
        "quarter",
        "status",
        "game_id",
    ]  # not including teams
    dfs = helpers.get_dfs()
    df = pd.concat(dfs)
    df = clean(df)

    df[to_nums] = df[to_nums].astype(np.float)
    df["quarter"] = df.quarter.astype(np.int8)

    for col in ml_cols:
        df[col] = df[col].map(calc.eq)

    # normdf.to_normed()
    # df = df.join(pd.get_dummies(df))
    # df = df.drop(to_dummies, axis=1)

    sports = list(df.sport.unique())
    sets = {}

    for sport in sports:
        print(f"sport: {sport}")
        sportdf = df[df.sport == sport]
        sportdf = sportdf.drop("sport", axis=1)
        g_ids = sportdf.game_id.unique()
        random.shuffle(g_ids)

        n = len(g_ids)
        idx = int(n * train_frac)

        tr_ids = g_ids[:idx]
        te_ids = g_ids[idx:]

        tr_df = sportdf[sportdf.game_id.isin(tr_ids)]
        te_df = sportdf[sportdf.game_id.isin(te_ids)]

        test_normed = normdf.norm_testset(te_df, tr_df, str_cols=to_dummies)
        train_normed = normdf.to_normed(tr_df, str_cols=to_dummies)

        test_qtrs = pd.get_dummies(test_normed.quarter, prefix="q")
        test_status = pd.get_dummies(test_normed.status)
        test_normed = test_normed.join([test_qtrs, test_status])
        test_normed = test_normed.drop(["status", "quarter"], axis=1)
        # test_normed = olutils.hot_teams(test_normed, cols=['h_team', 'a_team'])

        train_qtrs = pd.get_dummies(train_normed.quarter, prefix="q")
        train_status = pd.get_dummies(train_normed.status)
        train_normed = train_normed.join([train_qtrs, train_status])
        train_normed = train_normed.drop(["status", "quarter"], axis=1)
        # train_normed = olutils.hot_teams(train_normed, cols=['h_team', 'a_team'])

        sets[sport] = {"train": train_normed, "test": test_normed}

    return sets


def prep_loader(trainset, testset, model_name, device, batch_size=1, classify=False, shuffle=True):

    x, y = trainset[0].values()

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)

    writer = SummaryWriter(f"runs/{model_name}{time.asctime()}")
    if y.dim() == 0:
        y_shape = 1
    else:
        y_shape = y.shape[0]

    model = models.Model(in_dim=x.shape[0], out_dim=y_shape,
                  classify=classify).to(device)

    if classify:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters())  # , lr=0.0001)
    d = {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "criterion": criterion,
        "optimizer": optimizer,
        "model": model,
        "writer": writer,
        "classify": classify,
    }
    return d


if __name__ == "__main__":
    sets = main()

    print(sets)
