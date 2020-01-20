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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

PATH = "./one_liner.pth"
RUNNING_INTERVAL = 50


def train_epoch(d, epoch):
    print("training")
    d['model'].train()
    running_loss = 0.0
    for i, data in enumerate(d['train_loader'], 0):
        x, y = data["x"].to(device), data["y"].to(device)

        d['optimizer'].zero_grad()

        y_hat = d['model'](x.float())

        loss = d['criterion'](y_hat, torch.max(y, 1)[1])
        loss.backward()
        d['optimizer'].step()

        d['writer'].add_scalar("train_loss", loss, i +
                               epoch * len(d['train_loader']))
        d['writer'].add_scalar

        running_loss += loss.item()
        if i % RUNNING_INTERVAL == RUNNING_INTERVAL - 1:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
            print(f"y: {y}, y_hat: {y_hat}")

            running_loss = 0.0


def test_epoch(d, epoch):
    print("testing")
    d['model'].eval()

    running_loss = 0.0
    with torch.no_grad():
        for j, test_data, in enumerate(d['test_loader'], 0):
            test_x, test_y = test_data["x"].to(
                device), test_data["y"].to(device)

            test_y_hat = d['model'](test_x.float())
            test_loss = d['criterion'](test_y_hat, torch.max(test_y, 1)[1])

            d['writer'].add_scalar("test_loss", test_loss, j +
                                epoch * len(d['test_loader']))

            running_loss += test_loss.item()
            if j % RUNNING_INTERVAL == RUNNING_INTERVAL - 1:
                print(f"[{epoch + 1}, {j + 1}] loss: {running_loss / 2000}")
                print(f"test_y: {test_y}, test_y_hat: {test_y_hat}")
                running_loss = 0.0


def train():
    EPOCHS = 1
    BATCH_SIZE = 1
    CLASSIFY = True

    d = olutils.prep(classify=CLASSIFY, batch_size=BATCH_SIZE)

    running_loss = 0
    for epoch in range(EPOCHS):

        train_epoch(d, epoch)
        test_epoch(d, epoch)

        torch.save(d['model'].state_dict(), PATH)

    print("Finished Training")


def infer(fn='data.csv', preds_fn='data_preds.csv') -> pd.DataFrame:
    df = pd.read_csv(fn)

    state = torch.load('one_liner.pth')
    shapes = [l.shape for l in list(state.values())]

    m = olutils.Model(shapes[0][1], shapes[-1][0]).to(device)
    df = olutils.hot_teams(df)

    game_ids = df.pop('Game_id')
    tdf= torch.tensor(df.values, dtype=torch.float).to(device)
    cols = ['Game_id', 'H_win', 'A_win']
    m.eval()
    rows = []
    optimizer = optim.Adam(m.parameters(), lr=0.001)
    with torch.no_grad():
        for i, g_id in enumerate(game_ids):
            optimizer.zero_grad()
            yhat = m(tdf[i].view(1, -1)).cpu().numpy()
            rows.append([g_id, yhat[0][0], yhat[0][1]])

    preds = pd.DataFrame(rows, columns=cols)

    if preds_fn:
        preds.to_csv(preds_fn)
    
    return preds



if __name__ == "__main__":
    train()
    infer()
