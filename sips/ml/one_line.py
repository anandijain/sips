import torch

import pandas as pd
import numpy as np
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(196, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 125)
        self.fc4 = nn.Linear(125, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
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


def load_data(batch_size=1, verbose=False):
    df, df2 = [pd.read_csv(DIR + f) for f in FILES]
    m = df.merge(df2, left_on='Game_id', right_on='Game_id', how='outer')
    m = m.fillna(0)
    # target = pd.get_dummies(m.pop('H_win'))
    target = m.pop('H_win')
    m = m.select_dtypes(exclude=['object'])
    m = m.apply(pd.to_numeric, errors='coerce')
    # m = m.drop(['A_team_x', 'Game_id', 'H_team_x', 'A_plus_minus',
    # 'A_team_y', 'Arena', 'H_plus_minus', 'H_team_y', 'Time'], axis=1)
    # m = m[COLS]
    target.columns = ['H_win', 'A_win']
    m = (m-m.mean())/m.std()
    if verbose:
        print(f'xs: {m}, target: {target}')

    return m, target


def load_pretrained():
    net = Net()
    net.load_state_dict(torch.load(PATH))
    return net


if __name__ == "__main__":

    net = Net()
    dataset = OneLiner()
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/one_liner0')
    # writer.add_graph(net, images)
    # writer.close()
    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample['x'], sample['y'])
        if i == 3:
            x_shape = sample['x'].shape
            y_shape = sample['y'].shape
            break

    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['x'], data['y']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.reshape(1, -1).float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar('train_loss', loss, i + epoch*len(dataloader))
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), PATH)
