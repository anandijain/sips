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
        self.layers = []
        while cur != 1:
            cur = math.ceil(prev. / 2.)
            self.layers.append(nn.Linear(prev, cur))
            prev = cur

    def forward(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        return x

if __name__ == '__main__':
    df = pd.read_csv('./data/nba2.csv')
    grouped = [elt[1] for elt in df.groupby(['game_id', 'quarter'])]
    
