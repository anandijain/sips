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
        self.predict_col = predict_column
        self.train_cols = train_columns

        df = pd.read_csv('./data/static/big_daddy2.csv')
        labels = df['h_win'].copy()
        df = df[df.Gen_Games > 4]
        x = df[self.train_cols].values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled, columns=self.train_cols)
        self.projections_frame = pd.concat((df, labels), axis=1).fillna(0)

    def __len__(self):
        return len(self.projections_frame)

    def __getitem__(self, index):
        x = torch.tensor(self.projections_frame.iloc[index][self.train_cols], dtype=torch.float)
        y = self.projections_frame.iloc[index][self.predict_col].values
        if y == 1:
            y = torch.tensor([0., 1.], dtype=torch.float)
        elif y == 0:
            y = torch.tensor([1., 0.], dtype=torch.float)
        else:
            y = torch.tensor([0., 0.], dtype=torch.float)

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
    train_columns = ['a_gen_a_avg_allowed', 'a_gen_a_avg_first_downs', 'a_gen_a_avg_fourth_down_conv_pct', 'a_gen_a_avg_fumbles', 'a_gen_a_avg_fumbles_lost', 'a_gen_a_avg_interceptions', 'a_gen_a_avg_ints_per_attempt', 'a_gen_a_avg_pass_completion_pct', 'a_gen_a_avg_pass_tds', 'a_gen_a_avg_pass_yards', 'a_gen_a_avg_pass_yards_per_attempt', 'a_gen_a_avg_penalties', 'a_gen_a_avg_penalty_yards', 'a_gen_a_avg_rush_tds', 'a_gen_a_avg_rush_tds_per_attempt', 'a_gen_a_avg_rush_yards', 'a_gen_a_avg_rush_yards_per_attempt', 'a_gen_a_avg_sack_yards','a_gen_a_avg_sacks', 'a_gen_a_avg_score', 'a_gen_a_avg_third_down_conv_pct', 'a_gen_a_avg_top', 'a_gen_a_avg_total_yards', 'a_gen_a_avg_turnovers', 'a_gen_a_rec', 'gen_avg_allowed', 'gen_avg_first_downs', 'gen_avg_fourth_down_conv_pct', 'gen_avg_fumbles','gen_avg_fumbles_lost', 'gen_avg_interceptions', 'gen_avg_ints_per_attempt', 'gen_avg_pass_comp_pct', 'gen_avg_pass_tds', 'gen_avg_pass_yards', 'gen_avg_pass_yards_per_attempt', 'gen_avg_penalties', 'gen_avg_penalty_yards', 'gen_avg_rush_tds_per_attempt', 'gen_avg_rush_yards', 'gen_avg_rush_yards_per_attempt', 'gen_avg_sack_yards', 'gen_avg_sacks', 'gen_avg_score', 'gen_avg_third_down_conv_pct', 'gen_avg_top', 'gen_avg_total_yards', 'gen_avg_turnovers', 'h_gen_h_avg_allowed', 'h_gen_h_avg_first_downs', 'h_gen_h_avg_fourth_down_conv_pct', 'h_gen_h_avg_fumbles', 'h_gen_h_avg_fumbles_lost', 'h_gen_h_avg_interceptions', 'h_gen_h_avg_ints_per_attempt', 'h_gen_h_avg_pass_completion_pct', 'h_gen_h_avg_pass_tds', 'h_gen_h_avg_pass_yards', 'h_gen_h_avg_pass_yards_per_attempt', 'h_gen_h_avg_penalties', 'h_gen_h_avg_penalty_yards', 'h_gen_h_avg_rush_tds', 'h_gen_h_avg_rush_tds_per_attempt', 'h_gen_h_avg_rush_yards', 'h_gen_h_avg_rush_yards_per_attempt', 'h_gen_h_avg_sack_yards', 'h_gen_h_avg_sacks', 'h_gen_h_avg_score', 'h_gen_h_avg_top', 'h_gen_h_avg_total_yards', 'h_gen_h_avg_turnovers', 'h_gen_h_rec']

    save_path = './models/w_l_classify.pt'

    dataset = WinSet(train_columns=train_columns)
    data_loader = DataLoader(dataset, batch_size=16)

    (x, y) = dataset[0]
    print(f'x shape: {x.shape}, y shape: {y.shape}')
    print(f'x: {x}, y shape: {y}')
    in_dim = x.shape[0]
    out_dim = y.shape[0]


    lr = 1e-3
    num_epochs = 1
    log_interval = 50

    total = 0
    correct = 0
    acc = 0

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
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)

                correct += (predicted == torch.argmax(y, dim= 1)).sum().item()
                acc = correct / total
                # print('Accuracy of the network on the test images: %d %%' % (100 * ))
                # if i != 0:
                #     correct += 1
                #     acc = correct / i

            if i % log_interval == 0:
                with torch.no_grad():
                    print("Epoch ", epoch, "CrossEntropyLoss: ", loss.item())
                    print(f'y_pred: {y_pred}, y: {y}')
                    print(f'acc: {acc}')

        torch.save(model.state_dict(), save_path)
