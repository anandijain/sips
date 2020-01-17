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
device = torch.device('cpu')
print(device)

PATH = './one_liner.pth'

FILES = ['/home/sippycups/absa/sips/data/nba/nba_history.csv',
         '/home/sippycups/absa/sips/data/nba/nba_history_with_stats.csv']


COLS = ['Home_team_gen_avg_3_point_attempt_rate',
        'Home_team_gen_avg_3_pointers_attempted',
        'Home_team_gen_avg_3_pointers_made',
        'Home_team_gen_avg_assist_percentage',
        'Home_team_gen_avg_assists',
        'Home_team_gen_avg_block_percentage',
        'Home_team_gen_avg_blocks',
        'Home_team_gen_avg_defensive_rating',
        'Away_team_gen_avg_3_point_attempt_rate',
        'Away_team_gen_avg_3_pointers_attempted',
        'Away_team_gen_avg_3_pointers_made',
        'Away_team_gen_avg_assist_percentage',
        'Away_team_gen_avg_assists',
        'Away_team_gen_avg_block_percentage',
        'Away_team_gen_avg_blocks',
        'Away_team_gen_avg_defensive_rating',
        'Away_team_gen_avg_defensive_rebound_percentage',
        'Away_team_gen_avg_defensive_rebounds']


class Model(nn.Module):
    def __init__(self, in_dim, out_dim, classify=True):
        super(Model, self).__init__()
        self.classify = classify
        self.fc1 = nn.Linear(in_dim, in_dim-1)
        self.fc2 = nn.Linear(in_dim-1, 500)
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

    def __init__(self, data=None, norm=True, shuffle=True, verbose=True):
        """

        """
        if not data:
            xs, ys = load_data(norm=norm, shuffle=shuffle, verbose=verbose)

            self.xs = torch.tensor(xs.values)
            self.ys = torch.tensor(ys.values, dtype=torch.long)
        else:
            self.xs = torch.tensor(data[0].values)
            self.ys = torch.tensor(data[1].values)

        self.length = len(self.xs)

        if verbose:
            print(self.xs)
            print(self.ys.shape)
            print(self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # x = self.xs.iloc[idx].values
        return {'x': self.xs[idx], 'y': self.ys[idx]}


def get_data(fn):
    X = pd.read_csv(fn)
    
    X = X.select_dtypes(exclude=['object'])
    X = X.apply(pd.to_numeric, errors='coerce')

    hot.hot(X, nba.teams)

    return X

def load_data(train_file, predict_file=None, classify=False, batch_size=1, norm=True, verbose=False):
    """

    """
    X = pd.read_csv(train_file)


    X = X.fillna(0)


    if predict_file:
        stats = pd.read_csv(predict_file)
        if classify:
            target = stats['H_win']
            target.columns = ['H_win', 'A_win']
    else:
        to_pred = list(stats.columns)
        for item in ['A_team', 'H_team', 'Date', 'Season']:
            to_pred.remove(item)
        X = X.merge(stats, left_on='Game_id', right_on='Game_id', how='outer')
        print(X)
        # X = X.select_dtypes(exclude=['object'])
        X = X.apply(pd.to_numeric, errors='coerce')

        target = X[to_pred].copy()
        del X[to_pred]

    X = X.select_dtypes(exclude=['object'])
    X = X.apply(pd.to_numeric, errors='coerce')

    if norm:
        X = (X-X.mean())/X.std()

    if verbose:
        print(f'xs: {X}, target: {target}')

    return X, target


def prep(batch_size=1, classify=False, verbose=False):
    """

    """
    
    data = load_data(FILES, norm=False, classify=classify)
    dataset = OneLiner(data=data)

    if verbose:
        for i in range(len(dataset)):
            sample = dataset[i]

            print(i, sample['x'], sample['y'])
            if i == 0:
                x_shape = sample['x'].shape
                y_shape = sample['y'].shape
                break
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(f'runs/one_liner_{time.asctime()}')
    
    in_dim = len(dataset[0]['x'])
    if classify:
        out_dim = 2
    else:
        out_dim = len(dataset[0]['y'])

    model = Model(in_dim, out_dim)
    split_idx = len(dataset) // 2
    test_len = len(dataset) - split_idx

    train_set, test_set = torch.utils.data.random_split(
        dataset, [split_idx, test_len])


    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4)

    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=True, num_workers=4)

    writer = SummaryWriter(f'runs/one_liner_{time.asctime()}')

    if classify:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    for i, data in enumerate(train_loader, 0):
        x, y = data['x'].to(device), data['y'].to(device)
        optimizer.zero_grad()
        y_hat = model(x.reshape(1, -1).float())
        loss = criterion(y_hat, y)
        break

    d = {
        'dataset': dataset,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'model': model,
        'writer': writer,
        'x': x,
        'y': y,
        'y_hat': y_hat,
        'loss': loss
    }
    return d


if __name__ == "__main__":
    BATCH_SIZE = 1

    d = prep(classify=True, batch_size=BATCH_SIZE)
    
    dataset = d['dataset']
    train_loader = d['train_loader']
    test_loader = d['test_loader']
    model = d['model']
    writer = d['writer']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0

    for epoch in range(2):

        running_loss = 0.0
        print('training')
        model.train()
        for i, data in enumerate(train_loader, 0):
            x, y = data['x'].to(device), data['y'].to(device)

            optimizer.zero_grad()

            y_hat = model(x.reshape(BATCH_SIZE, -1).float())

            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            writer.add_scalar('train_loss', loss, i + epoch*len(train_loader))
            writer.add_scalar

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')
                print(f'y: {y}, y_hat: {y_hat}')

                running_loss = 0.0

        print('testing')
        model.eval()

        running_loss = 0.0
        for j, test_data, in enumerate(test_loader, 0):
            test_x, test_y = test_data['x'].to(
                device), test_data['y'].to(device)

            optimizer.zero_grad()

            test_y_hat = model(test_x.reshape(1, -1).float())
            test_loss = criterion(test_y_hat, test_y)

            writer.add_scalar('test_loss', test_loss,
                              j + epoch*len(test_loader))

            if j % 2000 == 1999:
                print(f'[{epoch + 1}, {j + 1}] loss: {running_loss / 2000}')
                print(
                    f'test_y: {test_y}, test_y_hat: {test_y_hat}')
                running_loss = 0.0

    print('Finished Training')

    torch.save(model.state_dict(), PATH)
