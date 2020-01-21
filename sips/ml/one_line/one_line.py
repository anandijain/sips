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
RUNNING_INTERVAL = 200


def train_epoch(d, epoch):
    print("training")
    d['model'].train()
    running_loss = 0.0
    for i, data in enumerate(d['train_loader'], 0):
        x, y = data["x"].to(device), data["y"].to(device)

        d['optimizer'].zero_grad()

        y_hat = d['model'](x)

        loss = d['criterion']((y_hat), torch.max(y, 1)[1])
        # loss = d['criterion'](y_hat, y)
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

CUTOFF = 0.65 # one class pred must be < 0.65
def test_epoch(d, epoch):
    print("testing")
    d['model'].eval()

    running_loss = 0.0
    with torch.no_grad():
        for i, test_data, in enumerate(d['test_loader'], 0):
            test_x, test_y = test_data["x"].to(
                device), test_data["y"].to(device)

            test_max = torch.max(test_y_hat)
            test_y_hat = d['model'](test_x)
            test_loss = d['criterion'](test_y_hat, torch.max(test_y, 1)[1])

            pred_win = torch.argmax(test_y_hat)
            if  test_max > CUTOFF and pred_win == test_y:
                print(f'confident correct bet in test {test_x}')                
            d['writer'].add_scalar("test_loss", test_loss, i +
                                   epoch * len(d['test_loader']))

            running_loss += test_loss.item()
            if i % RUNNING_INTERVAL == RUNNING_INTERVAL - 1:
                # print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
                print(f"test_y: {test_y}, test_y_hat: {test_y_hat}")
                running_loss = 0.0


def train(prepped, epochs=10):

    running_loss = 0
    for epoch in range(epochs):

        train_epoch(prepped, epoch)
        test_epoch(prepped, epoch)

        torch.save(prepped['model'].state_dict(), PATH)

    print("Finished Training")


def infer(fn='data.csv', preds_fn='data_preds.csv') -> pd.DataFrame:
    df = pd.read_csv(fn)
    
    old = olutils.train_dataset(norm=False, hot=False)[0]
    
    normed_df = norm_inferset(df, old)
    df = olutils.hot_teams(normed_df)
    print(df)

    state = torch.load('one_liner.pth')
    shapes = [l.shape for l in list(state.values())]

    m = olutils.Model(shapes[0][1], shapes[-1][0]).to(device)

    game_ids = df.pop('Game_id')
    # tdf= torch.tensor(df.values, dtype=torch.float).to(device)
    tdf = torch.tensor(df.values, dtype=torch.float).to(device)
    cols = ['Game_id', 'H_win', 'A_win']
    m.eval()
    rows = []

    with torch.no_grad():
        for i, g_id in enumerate(game_ids):
            x = tdf[i].view(1, -1)
            yhat = m(x).cpu().numpy()
            print(f'{g_id}: {yhat}')
            rows.append([g_id, yhat[0][0], yhat[0][1]])

    preds = pd.DataFrame(rows, columns=cols)

    if preds_fn:
        preds.to_csv(preds_fn)

    return preds


def norm_inferset(to_norm, norm_base):
    str_cols = ['Game_id', 'A_team', 'H_team']
    norm_base = norm_base.drop(str_cols, axis=1)
    infer_strs = to_norm[str_cols]
    to_norm = to_norm.drop(infer_strs, axis=1)

    to_norm = (to_norm-norm_base.min())/(norm_base.max()-norm_base.min())

    normed_df = pd.concat([infer_strs, to_norm], axis=1)

    return normed_df


if __name__ == "__main__":
    BATCH_SIZE = 16
    CLASSIFY = True

    d = olutils.prep(classify=CLASSIFY, batch_size=BATCH_SIZE)
    train(d)
    infer()
