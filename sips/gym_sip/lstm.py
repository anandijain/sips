import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import preprocessing

save_path = './models/lstm.pt'

class FileLoader:
    def __init__(self, directory):
        self.files = os.listdir(directory)
        self.length = len(self.files)

    def __getitem__(self, index):
        df = pd.read_csv(directory + files[index])
        return df.iloc[:, 1:5].values

#         x = df.values #returns a numpy array
#         min_max_scaler = preprocessing.MinMaxScaler()
#         x_scaled = min_max_scaler.fit_transform(x)
#         data = x_scaled

    def __len__(self):
        return self.length

directory = "/mnt/c/Users/Anand/home/Programming/datasets/price-volume-data-for-all-us-stocks-etfs/Data/Stocks/"
fileset = FileLoader(directory)

class LSTMLoader(Dataset):
    def __init__(self, data):
        self.samples = []
        self.length = len(data)
        self.window_len = 1
        self.data = data
        self.get_data()

    def get_data(self):
        for i in range(1, self.length - (self.window_len + 1)):
            upper_idx = i + self.window_len
            x = torch.tensor(self.data[i - 1:upper_idx - 1, :]).view(1, 1, -1).float()
            y = torch.tensor(self.data[upper_idx, :]).view(1, 1, -1).float()
            self.samples.append((x, y))

    def __len__(self):
        return self.length -  (self.window_len + 1)

    def __getitem__(self, index):
        return self.samples[index]

data = fileset[1]
dataset = LSTMLoader(data)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
sample_x, sample_y = dataset[0]
print(f'sample_x shape: {sample_x.shape}, sample_y shape: {sample_y.shape}')

# Here we define our model as a class
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

model = LSTM(4, 10, batch_size=1, output_dim=4, num_layers=10)
model.load_state_dict(torch.load(save_path))

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################

num_epochs = 1
break_idx = 100
hist = np.zeros(num_epochs * len(dataset))

for t in range(num_epochs):
    # Clear stored gradient
    for data_idx, np_data in enumerate(fileset):

        if data_idx == break_idx:
            break

        dataset = LSTMLoader(np_data)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        model.hidden = model.init_hidden()
        torch.save(model.state_dict(), save_path)

        print(f"new stock: {data_idx}")
        try:
            for i, (x, y) in enumerate(data_loader):
                model.zero_grad()

                # Initialise hidden state
                # Don't do this if you want your LSTM to be stateful


                # Forward pass
                y_pred = model(x)

                loss = loss_fn(y_pred, y)
                if t % 100 == 0:
                    print("Epoch ", t, "MSE: ", loss.item())
                if t % 1000 == 0:
                    print(f'y_pred: {y_pred}, y: {y}')
                hist[t] = loss.item()

                # Zero out gradient, else they will accumulate between epochs
                optimiser.zero_grad()

                # Backward pass
                loss.backward()

                # Update parameters
                optimiser.step()
        except IndexError:
            continue



plt.plot(y_pred.detach().numpy(), label="Preds")
plt.plot(y_train.detach().numpy(), label="Data")
plt.legend()
plt.show()

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()

if __name__ == "__main__":
