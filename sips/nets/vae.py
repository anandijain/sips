'''
variational autoencoder for multiline games

want to split by pre game and in game too

todo scheme:

- get padded quarters
'''
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class Net(nn.Module):
    '''
    This network is defined recursively.
    |layers| ~ log_2(input_dim)
    '''
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.input_dim = input_dim

        self.encoder = []
        self.decoder = []

        self.instantiate_network()

        # enc =

        self.enc = nn.ModuleList(self.encoder)
        self.dec = nn.ModuleList(self.decoder)

    def instantiate_network(self):
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
            x = F.relu(layer(x))
        return x

    def __repr__(self):
        print(f'encoder: {self.enc}')
        print(f'decoder: {self.dec}')
        return 'network'


class Loader(Dataset):
    def __init__(self):
        self.df = pd.read_csv('./data/nba2.csv')
        self.cols = ['game_id', 'cur_time', 'quarter', 'secs', 'a_pts', 'h_pts',
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

        self.padded = nn.utils.rnn.pad_sequence(grouped, padding_value=pad)
        print(self.padded.shape)
        item = self.padded[0]

        self.length = len(item.flatten())

    def __getitem__(self, index):
        return self.padded[index].flatten()

    def __len__(self):
        return self.length


if __name__ == '__main__':

    data_loader = Loader()
    net = Net(data_loader.length).double()

    print(net)
    optimizer = optim.RMSprop(net.parameters(), lr=1e-3)
    epochs = 1

    net.train()

    for i in range(1, epochs + 1):
        for j, (x, _) in enumerate(data_loader):
            optimizer.zero_grad()

            encoded = net.forward(x, net.enc)
            decoded = net.forward(encoded, net.dec)

            loss = torch.abs(x - decoded)

            loss.backward()
            optimizer.step()
