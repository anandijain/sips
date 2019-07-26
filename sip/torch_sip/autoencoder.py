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


class VAE(nn.Module):
    def __init__(self, input_size):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_size, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

batch_size = 1
df = h.get_df(fn='./data/3_years_of_data.csv')
num_cols = df.shape[1]
# scaled = h.sk_scale(df)

train_df, test_df = h.train_test(df, train_pct=0.3)

train = h.DfPastGames(train_df)
test = h.DfPastGames(test_df)

input_size = train.data_shape

train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

autoencoder = VAE(input_size)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

EPOCHS = 1

plt.ion()
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
                view_data = game.view(-1, input_size).type(torch.FloatTensor)/255.
                _, decoded_data = autoencoder(view_data)


plt.ioff()
plt.show()
