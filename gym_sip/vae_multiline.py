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

import h

class Net(nn.Module):
    def __init__(self, input_dim):
        prev = input_dim
        cur = input_dim
        tuples = []
        self.encoder = []
        self.decoder = []
        while cur != 1:
            cur = math.ceil(prev. / 2.)
            tuples.append(prev, cur)
            prev = cur

        for tup in tuples:
            self.encoder.append(nn.Linear(tup))

        for tup in tuples[::-1]:
            self.decoder.append(nn.Linear(tup[::-1]))

    def forward(self, x, direction):
        for layer in direction:
            x = nn.relu(layer(x))
        return x

if __name__ == '__main__':
    df = pd.read_csv('./data/nba2.csv')
    group = df.groupby(['game_id', 'quarter'])

    # might be a torch fxn to find max seq len
    max = 0
    grouped = []
    for elt in group:
        cur_len = len(elt)

        if cur_len > max:
            max = cur_len

        grouped.append(torch.tensor(elt[1].values, dtype=torch.double))

    pad = 0
    if max % 2 != 0:
        pad = 1

    padded = torch.utils.rnn.pad_sequence(t_grouped, pad_len=pad) # ideally multiple of 2

    net = Net(dim)

    optimizer = optim.RMSprop(net.parameters(), lr=1e-3)

    epochs = 1

    for i in range(1, epochs + 1):
        for j, pt in enumerate(data_loader):
            optimizer.zero_grad()
            x = pt[0]
            y = pt[1]

            encoded = self.forward(x, net.encoder)
            decoded = self.forward(encoded, net.decoder)

            loss = torch.abs(y - decoded)

            loss.backward()
            optimizer.step()
