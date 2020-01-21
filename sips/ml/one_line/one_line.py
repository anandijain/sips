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
    correct = 0
    total = 0
    running_loss = 0.0
    for i, data in enumerate(d['train_loader'], 0):
        x, y = data["x"].to(device), data["y"].to(device)
        class_idxs = torch.max(y, 1)[1]

        # model stuff
        d['optimizer'].zero_grad()
        y_hat = d['model'](x)
        loss = d['criterion']((y_hat), class_idxs)
        loss.backward()
        d['optimizer'].step()

        # accuracy
        preds = torch.max(y_hat, 1)[1]
        batch_size = y.size(0)
        total += batch_size

        batch_correct = (preds == class_idxs).sum().item()
        correct += batch_correct

        d['writer'].add_scalar("train_loss", loss, i +
                               epoch * len(d['train_loader']))
        d['writer'].add_scalar("train_acc", batch_correct / batch_size, i +
                               epoch * len(d['train_loader']))
        running_loss += loss.item()

        if i % RUNNING_INTERVAL == RUNNING_INTERVAL - 1:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
            print(f"y: {y}, y_hat: {y_hat}")
            running_loss = 0.0

    print(f'train accuracy {(100 * correct / total):.2f} %')


CUTOFF = 0.65  # one class pred must be < 0.65


def test_epoch(d, epoch):
    print("testing")
    d['model'].eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for i, test_data, in enumerate(d['test_loader'], 0):
            test_x, test_y = test_data["x"].to(
                device), test_data["y"].to(device)

            test_y_hat = d['model'](test_x)

            _, predicted = torch.max(test_y_hat.data, 1)
            # = torch.max(test_y.data, 1)

            class_idx = torch.max(test_y, 1)[1]
            test_loss = d['criterion'](test_y_hat, class_idx)
            batch_size = test_y.size(0)
            total += batch_size
            batch_correct = (predicted == class_idx).sum().item()
            correct += batch_correct

            # if test_max > CUTOFF and pred_win == class_idx:
            #     print(f'confident correct bet in test' \
            #         f'pred: {test_y_hat} real: {test_y}')

            d['writer'].add_scalar("test_loss", test_loss, i +
                                   epoch * len(d['test_loader']))
            d['writer'].add_scalar("test_acc", batch_correct / batch_size, i +
                                   epoch * len(d['test_loader']))

            running_loss += test_loss.item()
            if i % RUNNING_INTERVAL == RUNNING_INTERVAL - 1:
                # print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}")
                print(f"test_y: {test_y}, test_y_hat: {test_y_hat}")
                running_loss = 0.0
    print(f'test accuracy {(100 * correct / total):.2f} %')


def train(prepped, epochs=10):

    running_loss = 0
    for epoch in range(epochs):

        train_epoch(prepped, epoch)
        test_epoch(prepped, epoch)

        torch.save(prepped['model'].state_dict(), PATH)

    print("Finished Training")


def infer(d, fn='data.csv', preds_fn='data_preds.csv') -> pd.DataFrame:
    df = pd.read_csv(fn)
    # old = olutils.train_dfs()
    # old = old.drop(["A_win", "H_win"], axis=1)
    old = d['train_df']
    old = old.drop(["A_win", "H_win"], axis=1)
    print(f'old : {old.shape}')
    print(f'df : {df.shape}')
    normed_df = olutils.norm_testset(df, old)

    df = olutils.hot_teams(normed_df)
    # print(df)

    state = torch.load('one_liner.pth')
    shapes = [l.shape for l in list(state.values())]

    m = olutils.Model(shapes[0][1], shapes[-1][0]).to(device)

    game_ids = df.pop('Game_id')
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


if __name__ == "__main__":
    BATCH_SIZE = 128
    CLASSIFY = True

    d = olutils.prep(classify=CLASSIFY, batch_size=BATCH_SIZE)
    train(d)
    infer(d)
