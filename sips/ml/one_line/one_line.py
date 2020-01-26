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

from sips.ml.one_line import olutils
from sips.ml import normdf
from sips.ml import train as training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

PATH = "./one_liner.pth"


def train(prepped, epochs=10):

    running_loss = 0
    for epoch in range(epochs):

        training.train_epoch(prepped, epoch)
        training.test_epoch(prepped, epoch)

        torch.save(prepped["model"].state_dict(), PATH)

    print("Finished Training")


def infer(d, fn="data.csv", preds_fn="data_preds.csv") -> pd.DataFrame:
    df = pd.read_csv(fn)
    old = d["train_df"]
    old = old.drop(["A_win", "H_win"], axis=1)

    print(f"old : {old.shape}")
    print(f"df : {df.shape}")

    normed_df = normdf.norm_testset(df, old)

    df = olutils.hot_teams(normed_df)

    state = torch.load("one_liner.pth")
    shapes = [l.shape for l in list(state.values())]

    m = olutils.Model(shapes[0][1], shapes[-1][0]).to(device)

    game_ids = df.pop("Game_id")
    tdf = torch.tensor(df.values, dtype=torch.float).to(device)
    cols = ["Game_id", "H_win", "A_win"]
    m.eval()
    rows = []

    with torch.no_grad():
        for i, g_id in enumerate(game_ids):
            x = tdf[i].view(1, -1)
            yhat = m(x).cpu().numpy()
            print(f"{g_id}: {yhat}")
            rows.append([g_id, yhat[0][0], yhat[0][1]])

    preds = pd.DataFrame(rows, columns=cols)

    if preds_fn:
        preds.to_csv(preds_fn)

    return preds


if __name__ == "__main__":
    BATCH_SIZE = 128
    CLASSIFY = True

    d = olutils.prep(classify=CLASSIFY, batch_size=BATCH_SIZE)
    train(d)
    infer(d)
