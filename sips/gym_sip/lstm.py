import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import preprocessing


class FileLoader:
    def __init__(self, directory):
        self.files = os.listdir(directory)
        self.length = len(self.files)
        self.dir = directory

    def __getitem__(self, index):
        df = pd.read_csv(self.dir + self.files[index])
        return df.iloc[:, 1:5].values

#         x = df.values #returns a numpy array
#         min_max_scaler = preprocessing.MinMaxScaler()
#         x_scaled = min_max_scaler.fit_transform(x)
#         data = x_scaled

    def __len__(self):
        return self.length


class LSTMLoader(Dataset):
    def __init__(self, data, batch_size=1, window_len=10, predict_window=1):
        self.samples = []
        self.length = len(data)
        self.window_len = window_len
        self.predict_window = predict_window
        # self.batch_size = batch_size
        self.data = data
        self.get_data()

    def get_data(self):
        for i in range(1, self.length - (self.window_len + 1)):
            upper_idx = i + self.window_len
            x = torch.tensor(self.data[i - 1:upper_idx - 1, :]).view(1, 1, -1).float()
            y = torch.tensor(self.data[upper_idx:upper_idx + self.predict_window, :]).view(1, 1, -1).float()
            self.samples.append((x, y))

    def __len__(self):
        return self.length - (self.window_len + 1)

    def __getitem__(self, index):
        return self.samples[index]


class PlayerDataset(Dataset):

	team_columns = ['a_gen_a_avg_allowed', 'a_gen_a_avg_first_downs'] #list of player columns
	def __init__(self, fn='./data/static/lets_try.csv', predict_columns =['pass_rating', 'pass_yds', 'rush_yds', 'rec_yds'], team_columns = ['a_gen_a_avg_allowed', 'a_gen_a_avg_first_downs']):

		self.projections_frame = pd.read_csv(fn)
		self.transform(self.projections_frame)
		self.team_columns = team_columns
		self.predict_columns = predict_columns
		self.window = 10

	def transform(self, df):
		self.projections_frame = self.projections_frame.drop_duplicates()
		dfs = self.projections_frame.groupby('playerid')
		bigcsv = []
		for i in dfs:
			df = i[1]
			length = len(df)
			df = df.sort_values(by=['age'])
			if df.pass_yds.astype(bool).sum(axis=0) > .4*length or df.rec_yds.astype(bool).sum(axis=0) > .4*length or df.rush_yds.astype(bool).sum(axis=0) > .4*length:
				bigcsv.append(df)
		self.projections_frame = pd.concat(bigcsv)

	def __len__(self):
		return len(self.projections_frame)

	def __getitem__(self, index):
		past = torch.tensor(np.array(self.projections_frame[index:index+self.window][self.predict_columns + self.team_columns]))
		past = torch.tensor(past.reshape(-1, 1))
		team_data = torch.tensor(self.projections_frame.iloc[index+self.window][self.team_columns])
		# past = past.t()
		y = torch.tensor(self.projections_frame.iloc[index+self.window][self.predict_columns])
		past = past.float()
		team_data = team_data.float()
		x = torch.cat((past.flatten(), team_data.flatten())).view(1, 1, -1)

		tup = (x, y)
		return tup
#

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)



if __name__ == "__main__":

    save_path = './models/lstm.pt'
    directory = "/mnt/c/Users/Anand/home/Programming/datasets/price-volume-data-for-all-us-stocks-etfs/Data/Stocks/"

    fileset = FileLoader(directory)
    data = fileset[1]

    # dataset = LSTMLoader(data)
    dataset = PlayerDataset()
    sample_x, sample_y = dataset[0]
    print(f'sample_x shape: {sample_x.shape}, sample_y shape: {sample_y.shape}')

    input_shape = len(sample_x.flatten())
    output_shape = 4
    batch_size = 1
    hidden_shape = 10
    num_layers = 1

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for (x, y) in data_loader:
        break

    print(f'x shape: {x.shape}, y shape: {y.shape}')

    model = LSTM(input_shape, hidden_dim=hidden_shape, batch_size=batch_size, output_dim=output_shape, num_layers=num_layers)

    try:
        model.load_state_dict(torch.load(save_path))
        print(f'model state dict loaded from: {save_path}')
    except Exception:
        pass

    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 1
    break_idx = 100
    hist = np.zeros(num_epochs * len(dataset))

    for t in range(num_epochs):
        # Clear stored gradient
        for data_idx, np_data in enumerate(fileset):

            if data_idx == break_idx:
                break

            dataset = LSTMLoader(np_data)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            torch.save(model.state_dict(), save_path)
            model.hidden = model.init_hidden()
            print(f'new stock: {data_idx}')

            try:
                for i, (x, y) in enumerate(data_loader):
                    model.zero_grad()

                    x = x.view(1, batch_size, -1)

                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)

                    if t % 100 == 0:
                        print("Epoch ", t, "MSE: ", loss.item())
                    if t % 1000 == 0:
                        print(f'y_pred: {y_pred}, y: {y}')

                    hist[t] = loss.item()
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

            except IndexError:
                continue


    torch.save(model.state_dict(), save_path)

    plt.plot(y_pred.detach().numpy(), label="Preds")
    plt.plot(y_train.detach().numpy(), label="Data")
    plt.legend()
    plt.show()

    plt.plot(hist, label="Training loss")
    plt.legend()
    plt.show()
