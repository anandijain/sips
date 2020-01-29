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
import sips.ml.data_loaders as dls
from sips.ml import train

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RUNNING_INTERVAL = 200
PATH = "scores.pth"


def prep_loader():
    FIRST_N = 10
    LAST_N = 2
    # MIN_LEN = 150
    BATCH_SIZE = 1

    train_df, test_df = dls.normed_scoresets()
    trainset = dls.Scoreset(train_df, first_n=FIRST_N, last_n=LAST_N)
    testset = dls.Scoreset(test_df, first_n=FIRST_N, last_n=LAST_N)

    x, y = trainset[0].values()
    print(f'x{x}')
    print(f'x{x.shape[0]}')

    print(f'y{y}')
    print(f'y{y.shape[0]}')

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE)

    writer = SummaryWriter(f"runs/scores{time.asctime()}")

    model = train.Model(in_dim=x.shape[0], out_dim=y.shape[0]).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())  # , lr=0.0001)
    d = {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "criterion": criterion,
        "optimizer": optimizer,
        "model": model,
        "writer": writer,
        "classify": False,
    }
    return d


if __name__ == "__main__":
    d = prep_loader()
    train.train(d, 'scores', epochs=5)
