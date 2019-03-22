
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pandas as pd
import helpers as h
from torch.utils.data import Dataset, DataLoader


"""
Thanks to https://morvanzhou.github.io/tutorials/
https://www.youtube.com/user/MorvanZhou
"""


headers = [# 'a_team', 'h_team', 'sport', 'league', 
                'game_id', 'cur_time',
                'a_pts', 'h_pts', 'secs', 'status', 'a_win', 'h_win', 'last_mod_to_start',
                'num_markets', 'a_odds_ml', 'h_odds_ml', 'a_hcap_tot', 'h_hcap_tot']

# headers = ['game_id', 'order',  'a_pts',  'a_team', 'date', 'h_team', 'h_pts']

class Df(Dataset):
    def __init__(self, np_df, unscaled):
        self.data_len = len(np_df)
        self.data = np_df
        self.unscaled_data = unscaled
        print(self.data_len)

    def __getitem__(self, index):
        # line = self.data.iloc[index]
        line = self.data[index]
        line_tensor = torch.tensor(line)
        unscaled_line = self.unscaled_data[index]
        unscaled_tensor = torch.tensor(unscaled_line)        
        return line_tensor, unscaled_tensor

    def __len__(self):
        return self.data_len


def read_csv(fn='data/2019_szn'):  # read csv and scale data
    raw = pd.read_csv(fn + '.csv', usecols=headers)
    raw = raw.dropna()
    raw = pd.get_dummies(data=raw, columns=[ 'a_team', 'h_team', 'league', 'sport'])
    # raw = pd.get_dummies(data=raw, columns=[ 'a_team', 'h_team',])
    # raw = raw.drop(['game_id', 'lms_date', 'lms_time'], axis=1)
    raw = raw.drop(['game_id'], axis=1)
    print(raw.columns)
    # raw = raw.astype(np.float32)
    # raw = raw.sort_values('cur_time', axis=0)
    return raw.copy()


EPOCH = 1 
LR = 0.005
N_TEST_IMG = 5

tmp_df = h.get_df('./data/nba2.csv')
tmp_df = h.select_dtypes(tmp_df)
print(tmp_df.dtypes)

train_df, test_df = h.train_test(tmp_df)

scaled_train = h.sk_scale(train_df)
scaled_test = h.sk_scale(test_df)

train = h.Df(scaled_train, train_df.values)
test = h.Df(scaled_test, test_df.values)

batch_size = 1
num_cols = tmp_df.shape[1]

input_size = num_cols

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)


class AutoEncoder(nn.Module):
    def __init__(self, batch_size):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_cols * num_rows, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, ),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 10),
            # nn.max()

        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, num_cols),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder(batch_size)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

# initialize figure
f, a = plt.subplots(3, N_TEST_IMG, figsize=(9, 10))
plt.ion()   # continuously plot

print(train.data)
lt, ut = train.__getitem__(0)

view_data = lt.view(-1, num_cols).type(torch.FloatTensor)/255.

for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.numpy(), (9, 10)), cmap='brg'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

cur_data = torch.randn(1, num_cols).float()

num_games = 100

for game in range(num_games):
    for step, (x, unscaled) in enumerate(train_loader):
        prev_data = cur_data
        cur_data = x.float()
        # print('prev: {}'.format(prev_data))
        # print('cur: {}'.format(cur_data))

        encoded, decoded = autoencoder(prev_data)
        print(encoded)
        loss = loss_func(decoded, cur_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            with torch.no_grad():
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
                view_data = prev_data.view(-1, 90).type(torch.FloatTensor)/255.
                # plotting decoded image (second row)
                _, decoded_data = autoencoder(view_data)
                for i in range(N_TEST_IMG):
                    a[1][i].clear()
                    a[1][i].imshow(np.reshape(decoded_data.numpy(), (9, 10)), cmap='brg')
                    a[1][i].set_xticks(()); a[1][i].set_yticks(())
                    
                    a[2][i].clear()
                    a[2][i].imshow(np.reshape(cur_data.numpy(), (9, 10)), cmap='brg')
                    a[2][i].set_xticks(()); a[2][i].set_yticks(())
                plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()


torch.save(autoencoder.state_dict(), 'models/auto.ckpt')

view_data = torch.tensor(train.data[:200]).view(-1, 10).type(torch.FloatTensor)/255.
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2); ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train.data[:200]
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
plt.show()