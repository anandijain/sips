import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import helpers as h

torch.manual_seed(1)

df = h.get_df()
loader = h.Df(df)

TARGET_SIZE = loader.next_n
EMBEDDING_DIM = 99
HIDDEN_DIM = 99


class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_size):
        super(LSTMTagger, self).__init__()
        
        self.hidden_dim = hidden_dim
        # shape is 3D (sequence, indexes instances in mini-batch, indexes inputs)
        # x_t = (500, 1, 99)
        # h_{t-1} = (500, 1, 99)  ?
        self.lstm = nn.LSTM(input_dim, hidden_dim)

        self.hidden2target = nn.Linear(hidden_dim, target_size)

    def forward(self, tensors):
        lstm_out, _ = self.lstm(tensors.view(len(tensors), 1, -1))

        target = self.hidden2target(lstm_out.view(len(tensors), -1))

        prediction = F.log_softmax(target, dim=1)

        return prediction


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, TARGET_SIZE)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


for epoch in range(2):
    for num, data in enumerate(loader):

        if data is None:
            break

        model.zero_grad()

        train = data[0].float()
        targets = data[1].float()

        out = model(train)

        loss = loss_function(out, targets)

        loss.backward()
        optimizer.step()

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
