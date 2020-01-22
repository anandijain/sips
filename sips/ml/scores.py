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
from sips.ml import train as training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RUNNING_INTERVAL = 200
PATH = "scores.pth"

class Model(nn.Module):
    def __init__(self, in_dim, out_dim, classify=True):
        super(Model, self).__init__()
        self.classify = classify
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc4(x)
        x = F.relu(self.fc6(x))
        return x




def train(d, epochs=10):

    for epoch in range(epochs):
        training.train_epoch(d, epoch=epoch, verbose=True)
        torch.save(d['model'].state_dict(), PATH)



def prep_loader():
    FIRST_N = 100
    LAST_N = 50
    MIN_LEN = 150
    SPORT = 'BASK'
    BATCH_SIZE = 1

    fs = dls.Scoreset(first_n=FIRST_N, last_n=LAST_N,
                     min_len=MIN_LEN, sport=SPORT)

    x, y = fs[0].values()
    train_loader = DataLoader(fs, batch_size=BATCH_SIZE)

    writer = SummaryWriter(f"runs/scores{time.asctime()}")

    model = Model(in_dim=x.shape[0], out_dim=y.shape[0]).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    d = {
        "train_loader": train_loader,
        # "test_loader": test_loader,
        "criterion": criterion,
        "optimizer": optimizer,
        "model": model,
        "writer": writer,
        'classify': False,
    }
    return d



if __name__ == "__main__":
    d = prep_loader()
    train(d)
