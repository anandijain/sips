import time
import pandas as pd
import numpy as np

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

PATH = './one_liner.pth'
DIR = '/home/sippycups/absa/sips/data/nba/'
FILES = ['nba_history_with_stats.csv', 'nba_history.csv']
'/home/sippycups/absa/sips/data/nba/nba_history_with_stats.csv'

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
    def __init__(self, dim, classify=True):
        super(Model, self).__init__()
        self.classify = classify
        self.fc1 = nn.Linear(dim, dim-1)
        self.fc2 = nn.Linear(dim-1, 500)
        self.fc3 = nn.Linear(500, 250)
        self.fc4 = nn.Linear(250, 100)
        self.fc5 = nn.Linear(100, 10)
        self.fc6 = nn.Linear(10, 2)
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

    def __init__(self, data=None):
        """

        """
        if not data:
            xs, ys = load_data(verbose=True)
        print(xs)
        self.xs = torch.tensor(xs.values)
        # print(self.xs)
        self.ys = torch.tensor(ys.values, dtype=torch.long)
        print(self.ys.shape)
        self.length = len(self.xs)
        print(self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # x = self.xs.iloc[idx].values
        return {'x': self.xs[idx], 'y': self.ys[idx]}


def load_data(batch_size=1, norm=True, verbose=False):
    """

    """
    df, df2 = [pd.read_csv(DIR + f) for f in FILES]
    m = df.merge(df2, left_on='Game_id', right_on='Game_id', how='outer')
    m = m.fillna(0)
    target = m.pop('H_win')

    m = m.select_dtypes(exclude=['object'])
    m = m.apply(pd.to_numeric, errors='coerce')

    target.columns = ['H_win', 'A_win']

    if norm:
        m = (m-m.mean())/m.std() 

    if verbose:
        print(f'xs: {m}, target: {target}')

    return m, target


def prep():

    dataset = OneLiner()
    dim = len(dataset[0]['x'])
    model = Model(dim)
    split_idx = len(dataset) // 2
    test_len = len(dataset) - split_idx

    train_set, test_set = torch.utils.data.random_split(
        dataset, [split_idx, test_len])

    # default `log_dir` is "runs" - we'll be more specific here

    writer = SummaryWriter(f'runs/one_liner_{time.asctime()}')

    train_loader = DataLoader(train_set, batch_size=1,
                              shuffle=True, num_workers=4)

    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=True, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for i, data in enumerate(train_loader, 0):
        x, y = data['x'], data['y']
        optimizer.zero_grad()
        y_hat = model(x.reshape(1, -1).float())
        loss = criterion(y_hat, y)
        break

    d = {
        'dataset' : dataset,
        'train_set': train_set,
        'test_set' : test_set,
        'model': model,
        'x' : x,
        'y' : y,
        'y_hat' : y_hat,
        'loss': loss
    }
    return d



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)
    dataset = OneLiner()
    dim = len(dataset[0]['x'])
    model = Model(dim)
    model.to(device)
    split_idx = len(dataset) // 2
    test_len = len(dataset) - split_idx 

    train_set, test_set = torch.utils.data.random_split(dataset, [split_idx, test_len]) 

    # default `log_dir` is "runs" - we'll be more specific here

    writer = SummaryWriter(f'runs/one_liner_{time.asctime()}')
    
    train_loader = DataLoader(train_set, batch_size=1,
                            shuffle=True, num_workers=4)
    
    test_loader = DataLoader(test_set, batch_size=1,
                            shuffle=True, num_workers=4)

    Xs, Ys = next(iter(train_loader))
    
    # writer.add_graph(model, Xs)
    # writer.close()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample['x'], sample['y'])
        if i == 0:
            x_shape = sample['x'].shape
            y_shape = sample['y'].shape
            break

    for epoch in range(2):

        running_loss = 0.0
        print('training')
        model.train()
        for i, data in enumerate(train_loader, 0):
            x, y = data['x'].to(device), data['y'].to(device)

            optimizer.zero_grad()

            outputs = model(x.reshape(1, -1).float())
            
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            writer.add_scalar('train_loss', loss, i + epoch*len(train_loader))
            writer.add_scalar

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        print('testing')
        model.eval()
        for j, test_data, in enumerate(test_loader, 0):
            test_x, test_y = test_data['x'].to(device), test_data['y'].to(device)
            optimizer.zero_grad()

            outputs = model(test_x.reshape(1, -1).float())
            test_loss = criterion(outputs, test_y)

            writer.add_scalar('test_loss', test_loss, j + epoch*len(test_loader))
            if j % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, j + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    torch.save(model.state_dict(), PATH)
