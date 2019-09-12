import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sips.gym_sip import h


class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_size):
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)

        self.hidden2target = nn.Linear(hidden_dim, target_size)

    def forward(self, tensors):
        lstm_out, _ = self.lstm(tensors.view(len(tensors), 1, -1))

        target = self.hidden2target(lstm_out.view(len(tensors), -1))

        prediction = F.log_softmax(target, dim=1)

        return prediction


if __name__ == "__main__":

    batch_size = 128


    df = h.get_df()

    train = h.Df(df)

    item = train.__getitem__(500)

    input_size = h.num_flat_features(item[0])
    output_size = h.num_flat_features(item[1])
    hidden_size = (input_size + output_size) // 2

    TARGET_SIZE = input_size  # (1, 5, 99)
    EMBEDDING_DIM = 5 * input_size  # (1, 50, 99)
    HIDDEN_DIM = 5 * input_size  # (1, 50, 99)

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, TARGET_SIZE)
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)

    for epoch in range(2):
        for num, data in enumerate(train_loader):

            if data is None:
                break

            model.zero_grad()

            train = data[0].float()
            targets = data[1].float()


            out = model(train)


            print(train[0])
            print(targets[0])
            print(out[0])
            loss = loss_function(out, targets)

            print('loss: {}'.format(loss))

            loss.backward()
            optimizer.step()
