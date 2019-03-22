import torch
import torch.nn as nn
import torch.utils.data as Data

import matplotlib.pyplot as plt

from matplotlib import cm
import numpy as np
import pandas as pd
import helpers as h
from torch.utils.data import Dataset, DataLoader

EPOCH = 1 
LR = 0.005

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: {}'.format(device))


class AutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 1),
            nn.Sigmoid(), 
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 2048),
            nn.Tanh(),
            nn.Linear(2048, input_size),
            nn.Sigmoid(), 
        )

    def forward(self, x):
        print(x.shape)
        encoded = self.encoder(x.float())
        decoded = self.decoder(encoded)
        print(decoded.shape)
        return encoded, decoded

df = h.get_df()
num_cols = df.shape[1]
train_df, test_df = h.train_test(df)
scaled_df = h.sk_scale(train_df)
games = h.get_t_games(scaled_df)
test_games = h.get_t_games(test_df)

train = h.DfGame(games)
test = h.DfGame(test_games)

input_size = num_cols * train.game_len // 2
batch_size = 1

train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

autoencoder = AutoEncoder(input_size)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

EPOCHS = 1

for i in range(EPOCHS):
    for step, game in enumerate(train_loader):
        game = game[0].float()
        game = torch.reshape(game, (-1, 1))
        game = torch.squeeze(game)

        encoded, decoded = autoencoder(game)
        print(encoded)
        decoded = decoded.reshape(-1, 1)
        decoded = torch.squeeze(decoded)
        loss = loss_func(decoded, game)

        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            with torch.no_grad():
                print('Epoch: ', i, '| train loss: %.4f' % loss.data.numpy())
                # view_data = game.view(-1, num_cols).type(torch.FloatTensor)/255.
                # _, decoded_data = autoencoder(view_data)


plt.ioff()
plt.show()
