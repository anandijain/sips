#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# import h
import h

# credit https://github.com/utkuozbulak/pytorch-custom-dataset-examples

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()

        self.l1 = nn.Linear(input_size, input_size * 4)
        self.l2 = nn.Linear(input_size * 4, hidden_size)
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
        x = self.fc3(x)

        return x.double()


if __name__ == "__main__":

    batch_size = 25
    train_years = ['2011', '2013', '2014', '2015', '2016', '2017']
    test_years = ['2018']
    sport = 'mlb'
    train_fns = ['./data/' + year + sport + '.csv' for year in train_years]
    test_fns = ['./data/' + year + sport + '.csv' for year in test_years]
    def clean(fn):
        df = pd.read_csv(fn)
        df = df.dropna()
        df = df.drop_duplicates()
        return df   

    def _eq(odd):
    # to find the adjusted odds multiplier 
    # returns float
        if odd == 0:
            return 0
        if odd >= 100:
            return odd/100.
        elif odd < 100:
            return abs(100/odd) 


    def dfs(fns):
        dfs = []
        for fn in fns:
            tmp_df = clean(fn)
            dfs.append(tmp_df)
        df = pd.concat(dfs)
        return df


    
    train_df = dfs(train_fns)
    test_df = dfs(test_fns)
    num_cols = train_df.shape[1]
    print(len(train_df))
    # train_df, test_df = h.train_test(df, train_pct=0.7)
    feature_cols = ['Past_10_v', 'Past_10_h', 'rating1_pre', 'rating2_pre']
    train = h.DfCols(train_df, train_cols=feature_cols, label_cols=['h_win'])
    test = h.DfCols(test_df, train_cols=feature_cols, label_cols=['h_win'])

    # train = h.Df(train_df)
    # test = h.Df(test_df)

    item = train.__getitem__(500)
    print(item)
    # input_size = h.num_flat_features(item[0])
    # output_size = h.num_flat_features(item[1])

    input_size = len(train.data[0])
    output_size = len(train.labels[0])

    hidden_size = 16 # (input_size + output_size) // 2

    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

    net = Net(input_size, hidden_size, output_size)
    print(net)

    calc_loss = nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters())

    EPOCHS = 25
    steps = 0
    running_loss = 0
    correct = 0
    p_val = 5e-1

    probs = []
    home_mls = test_df['h_ML']
    home_mls = list(home_mls)
    away_mls = test_df['Open']
    away_mls = list(home_mls)
    homeresults = test_df['h_win']
    homeresults = list(homeresults)

    for epoch_num in range(EPOCHS):
        print("epoch: {} of {}".format(epoch_num, EPOCHS))
        for step_num, item in enumerate(train_loader):
            optimizer.zero_grad()

            data = item[0]
            target = item[1].double()
            pred = net(data)

            loss = calc_loss(pred, target)

            # plt_y = loss.detach()
            # plt_x = step_num * (epoch_num + 1)
            # plt.scatter(plt_x, plt_y, c='r', s=0.1)

            # with torch.no_grad():
            #     if step_num % 100 == 1:
            #         print('step: {}'.format(step_num))
            #         # print('input: {}'.format(data))
            #         print('pred: {}'.format(pred[0]))
            #         print('target: {}'.format(target[0]))
            #         print('loss: {}'.format(loss), end='\n\n')
            
            # running_loss += abs(loss)
            loss.backward()
            optimizer.step()


    # TESTING

    for test_step, test_item in enumerate(test_loader):

            test_data = test_item[0]
            target = test_item[1].double()
            with torch.no_grad():

                pred = net(test_data)
                #print(pred.shape)

                probs.append(pred)
                test_loss = calc_loss(pred, target)

                # if test_step % 10 == 1:
                #     print('step: {}'.format(step_num))
                    # print('input: {}'.format(test_data))
                    #print('pred: {}'.format(pred[0]))
                    #print('target: {}'.format(target[0]))

                if abs(test_loss) < p_val:
                    correct += 1        
    length = len(home_mls)
    n = 0

    for i in range(length):
        h_line = home_mls[i]
        a_line = away_mls[i]
        print(probs[i])
        home_winprob = probs[i][0]
        away_winprob = 1 - home_winprob
        home_result = homeresults[i]
        evhome = home_winprob * _eq(h_line) - away_winprob 
        evaway = away_winprob * _eq(a_line) - home_winprob
        evhome.tolist()
        evaway.tolist()
        print(evhome)
        print(evaway)
        if home_result == 1:
            roi_home = _eq(h_line)
            roi_away = -1

        if home_result == 0:
            roi_home = -1
            roi_away = _eq(a_line)

        if evaway > 0:
            n += roi_away
            print(n)
        if evhome > 0:
            n += roi_home
            print(n)
        plt.scatter(i, n, c='r', s=0.1)

            
#print('correct guesses: {} / total guesses: {}'.format(correct, test_step))
test_df
plt.show()

