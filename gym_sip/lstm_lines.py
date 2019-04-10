import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import helpers as h

torch.manual_seed(1)

df = h.get_df()

INPUT_SIZE = len(df)  # num columns
EMBEDDING_DIM = 64
HIDDEN_DIM = 64

df_loader = h.Df(df, 500, 5)

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, target_size):
        super(LSTMTagger, self).__init__()
        
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2target = nn.Linear(hidden_dim, target_size)

    def forward(self, tensors):
        lstm_out, _ = self.lstm(tensors.view(len(tensors), 1, -1))

        target = self.hidden2target(lstm_out.view(len(tensors), -1))

        prediction = F.log_softmax(target, dim=1)

        return prediction


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, INPUT_SIZE)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(2):
    for num, data in enumerate(custom_dataframe)

        model.zero_grad()
        train = data[0]
        targets = data[1]

        out = model(train)

        loss = loss_function(out, targets)

        loss.backward()
        optimizer.step()

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
