import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import helpers as h

# credit https://github.com/utkuozbulak/pytorch-custom-dataset-examples

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()

        self.l1 = nn.Linear(input_size, input_size * 2)
        self.l2 = nn.Linear(input_size * 2, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x.float()))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x.double()

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    batch_size = 1
    df = h.get_df(fn='./data/3_years_of_data.csv')
    num_cols = df.shape[1]
    # scaled = h.sk_scale(df)

    train_df, test_df = h.train_test(df, train_pct=0.3)

    train = h.DfPastGames(train_df)
    test = h.DfPastGames(test_df)

    input_size = train.data_shape
    output_size = 2

    hidden_size = 64

    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

    net = Net(input_size, hidden_size, output_size)
    print(net)

    #lr = 10

    calc_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())

    EPOCHS = 2
    steps = 0
    running_loss = 0
    correct = 0
    p_val = 1e-2

    for epoch_num in range(EPOCHS):

        for step_num, game in enumerate(train_loader):

            data = game[0]
            score_target = game[1].double()

            pred_score = net(data)
            loss = calc_loss(pred_score, score_target)
            plt_y = loss.detach()
            plt_x = step_num * (epoch_num + 1)
            plt.scatter(plt_x, plt_y, c='r', s=0.1)

            with torch.no_grad():
                if step_num % 10 == 1:
                    print('step: {}'.format(step_num))
                    print('pred_second: {}'.format(pred_score))
                    print('actual second half: {}'.format(score_target))
                    print('loss: {}'.format(loss), end='\n\n')
            
            running_loss += abs(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for test_step, test_game in enumerate(test_loader):
            test_data = test_game[0]
            score_target = test_game[1].double()
            with torch.no_grad():

                pred_score = net(test_data)
                loss = calc_loss(pred_score, score_target)


                if test_step % 10 == 1:
                    print('step: {}'.format(step_num))
                    print('pred_second: {}'.format(pred_score))
                    print('actual second half: {}'.format(score_target))
                    print('loss: {}'.format(loss), end='\n\n')

                if abs(loss) < 20:
                    correct += 1        

print('correct guesses: {} / total guesses: {}'.format(correct, test_step))
plt.show()

