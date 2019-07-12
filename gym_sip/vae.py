'''
variational autoencoder for multiline games

want to split by pre game and in game too

todo scheme:

- get padded quarters
'''
import math

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import h

'''
This network is defined recursively.
|layers| ~ log_2(input_dim)
'''
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.input_dim = input_dim

        self.encoder = []
        self.decoder = []

        # self.instantiate_network()

        prev = self.input_dim
        cur = self.input_dim

        tuples = []

        while cur != 1:
            cur = prev // 2
            tuples.append((prev, cur))
            prev = cur

        print(tuples)

        for tup in tuples:
            self.encoder.append(nn.Linear(tup[0], tup[1]))

        for tup in tuples[::-1]:
            self.decoder.append(nn.Linear(tup[1], tup[0]))


    def forward(self, x, direction):
        for layer in direction:
            x = nn.relu(layer(x))
        return x

    def __repr__(self):
        print(f'encoder: {str(self.encoder)}')
        print(f'decoder: {str(self.decoder)}')
        return 'network'


class Loader(Dataset):
    def __init__(self):
        self.df = pd.read_csv('./data/nba2.csv')
        self.cols = ['game_id, cur_time, quarter', 'secs', 'a_pts', 'h_pts',
                    'status', 'a_win', 'h_win', 'last_mod_to_start',
                    'last_mod_lines', 'num_markets', 'a_odds_ml', 'h_odds_ml',
                    'a_odds_ps', 'h_odds_ps', 'a_hcap_ps', 'h_hcap_ps',
                    'game_start_time']

        self.df_parsed = self.df[self.cols]
        group = self.df_parsed.groupby(['game_id', 'quarter'])

        # might be a torch fxn to find max seq len
        max = 0
        grouped = []
        for elt in group:
            cur_len = len(elt)

            if cur_len > max:
                max = cur_len
            grouped.append(torch.tensor(elt[1].values, dtype=torch.double))

        self.grouped = grouped

        pad = 0
        if max % 2 != 0:
            pad = 1

        self.padded = nn.utils.rnn.pad_sequence(t_grouped, padding_value=pad) # ideally multiple of 2
        self.length = len(self.padded)

    def __getitem__(self, index):
        return self.padded(index)

    def __len__(self):
        return self.length


if __name__ == '__main__':

    data_loader = Loader()

    net = Net(dim)
    print(net)

    optimizer = optim.RMSprop(net.parameters(), lr=1e-3)

    epochs = 1

    for i in range(1, epochs + 1):
        for j, pt in enumerate(data_loader):
            optimizer.zero_grad()
            x = pt[0]

            encoded = self.forward(x, net.encoder)
            decoded = self.forward(encoded, net.decoder)

            loss = torch.abs(x - decoded)

            loss.backward()
            optimizer.step()
