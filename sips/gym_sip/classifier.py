import os

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing

import matplotlib.pyplot as plt


class WinSet(Dataset):
    def __init__(self, predict_column = ['h_win'], train_columns = ['gen_avg_allowed', 'gen_avg_pass_comp_pct', 'gen_avg_pass_yards', 'gen_avg_rush_yards', 'gen_avg_rush_yards_per_attempt', 'gen_avg_score', 'gen_avg_total_yards']):
        self.projections_frame = pd.read_csv('./data/static/big_daddy2.csv')

        self.predict_col = predict_column
        self.train_cols = train_columns

    def __len__(self):
        return len(self.projections_frame)

    def __getitem__(self, index):
        x = torch.tensor(self.projections_frame.iloc[index][self.train_cols])
        y = self.projections_frame.iloc[index][self.predict_col].values
        if y == 1:
            y = torch.tensor([0., 1.], dtype=torch.double)
        elif y == 0:
            y = torch.tensor([1., 0.], dtype=torch.double)
        else:
            y = torch.tensor([0., 0.], dtype=torch.double)

        tup = (x, y)
        return tup


def log_dims(input_dim=784, output_dim=10, factor=2):
    '''
    mnist mlp w factor 2:
    [784, 397, 203, 106, 58, 34, 22, 16, 13, 11, 10]
    '''
    dims = []
    dim = input_dim
    delta = input_dim - output_dim

    while dim > output_dim:
        dims.append(dim)
        dim = (delta // factor) + output_dim
        delta = dim - output_dim

    dims.append(output_dim)
    print(dims)
    return dims

def get_layers(layer_dims, layer_type):
    layers = []
    num_layers = len(layer_dims)
    for i in range(num_layers):
        if i == num_layers - 1:
            break
        layers.append(layer_type(layer_dims[i], layer_dims[i + 1]))
    return layers


def mlp_layers(layer_dims, verbose=False):
    layer_type = nn.Linear
    layers = get_layers(layer_dims, layer_type)
    if verbose:
        print(layers)
    return layers

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, factor=2):
        super(MLP, self).__init__()
        self.layers = mlp_layers(log_dims(input_dim, output_dim, factor=factor))
        self.num_layers = len(self.layers)
        print(f'num_layers: {self.num_layers}')
        self.model = nn.ModuleList(self.layers)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i == self.num_layers - 1:
                # print(f'sm on layer {i}')
                x = F.softmax(layer(x), dim=1)
                break
            x = torch.tanh(layer(x))
        return x

if __name__ == "__main__":
    save_path = './models/w_l_classify.pt'

    dataset = WinSet()
    data_loader = DataLoader(dataset)

    (x, y) = dataset[0]
    print(f'x shape: {x.shape}, y shape: {y.shape}')
    in_dim = x.shape[0]
    out_dim = y.shape[0]

    lr = 1e-3
    num_epochs = 1
    log_interval = 500

    correct = 0

    model = MLP(in_dim, out_dim)
    
    try:
        model.load_state_dict(torch.load(save_path))
        print(f'model state dict loaded from: {save_path}')
    except Exception:
        pass


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(data_loader):
            optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, torch.max(y, 1)[1])

            loss.backward()
            optimizer.step()

            if torch.argmax(y_pred) == torch.argmax(y):
                if i != 0:
                    correct += 1
                    acc = correct / i

            if i % log_interval == 0:
                with torch.no_grad():
                    print("Epoch ", epoch, "CrossEntropyLoss: ", loss.item())
                    print(f'y_pred: {y_pred}, y: {y}')
                    print(f'correct, acc: {acc}')

        torch.save(model.state_dict(), save_path)
