import torch
import torch.nn as nn
import torch.nn.functional as F     

import pandas as pd
import numpy as np

import sips.gym_sip.h.loaders as l

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()

        self.l1 = nn.Linear(in_dim, in_dim*2)
        self.l2 = nn.Linear(in_dim*2, in_dim*2)
        self.l3 = nn.Linear(in_dim*2, in_dim*2)
        self.l4 = nn.Linear(in_dim*2, out_dim)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = torch.tanh(self.l3(out))
        out = self.l4(out)
        return out

if __name__ == "__main__":
    load = False # saved in ./models
    path = './models/predict_odds.pt'

    save = True
    lr = 1e-3
    epochs = 30
    log_interval = 1000
    batch_size = 16
    
    df = pd.read_csv('./data/static/nba2.csv')
    df = df[(df.last_mod_to_start > 0) & (df.a_pts > 0) & (df.h_pts > 0)] 
    games = [game[1] for game in df.groupby('game_id')]

    dataset_obj = l.LineGen(df)
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_size=batch_size)
    
    example_x, example_y = dataset_obj[1]
    in_dim = len(example_x)
    out_dim = len(example_y)

    net = Net(in_dim, out_dim)
    if load:
        net.load_state_dict(torch.load(path))
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        for game in games:
            try:
                dataset_obj = l.LineGen(game)
                data_loader = torch.utils.data.DataLoader(dataset_obj, batch_size=batch_size)
                for i, (x, y) in enumerate(data_loader):
                    optim.zero_grad()
                    output = net(x)
                    loss = criterion(output, y)
                    loss.backward()
                    optim.step()
                    with torch.no_grad():
                        if i % log_interval == 0:
                            print(f'loss: {loss}')
                            print(f'input: {x[0]}')
                            print(f'out: {output[0]}, y: {y[0]}\n')
            except IndexError:
                continue
    if save:        
        torch.save(net.state_dict(),path)
