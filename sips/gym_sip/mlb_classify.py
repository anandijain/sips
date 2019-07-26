#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.optim as optim

import h

# credit https://github.com/utkuozbulak/pytorch-custom-dataset-examples

# MACROS
batch_size = 1
train_years = ['2013', '2014', '2015', '2018', '2016', '2011']
test_years = ['2017']
sport = 'mlb'
EPOCHS = 1


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()

        self.l1 = nn.Linear(input_size, input_size * 4)
        self.l2 = nn.Linear(input_size * 4, hidden_size)
        # self.l3 = nn.Linear(hidden_size, hidden_size)
        # self.l4 = nn.Linear(hidden_size, hidden_size)
        # self.l5 = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x.float()))
        x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        # x = F.relu(self.l4(x))
        # x = F.relu(self.l5(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x.double()

def clean(fn):
    df = pd.read_csv(fn)
    df = df.dropna()
    df = df.drop_duplicates()
    return df   

def dfs(fns):
    dfs = []
    for fn in fns:
        tmp_df = clean(fn)
        dfs.append(tmp_df)
    df = pd.concat(dfs)
    return df


if __name__ == "__main__":

    train_fns = ['./data/' + year + sport + '.csv' for year in train_years]
    test_fns = ['./data/' + year + sport + '.csv' for year in test_years]

    train_df = dfs(train_fns)
    test_df = dfs(test_fns)
    
    num_cols = train_df.shape[1]

    print('training on : {} lines and {} columns'.format(len(train_df), num_cols))

    feature_cols = ['Past_10_v', 'Past_10_h', 'rating1_pre', 'rating2_pre', 'v_win', 'h_ML']

    train = h.DfCols(train_df, train_cols=feature_cols, label_cols=['h_win'])
    test = h.DfCols(test_df, train_cols=feature_cols, label_cols=['h_win'])

    input_size = len(train.data[0])
    output_size = len(train.labels[0])

    hidden_size = 16 # (input_size + output_size) // 2

    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

    net = Net(input_size, hidden_size, output_size)
    print(net)

    calc_loss = nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters())

    steps = 0
    running_loss = 0
    correct = 0
    p_val = .1

    probs = []
    home_mls = test_df['h_ML']
    home_mls = list(home_mls)
    away_mls = test_df['Open']
    away_mls = list(away_mls)
    homeresults = test_df['h_win']
    homeresults = list(homeresults)
    awayresults = test_df['v_win']
    awayresults = list(awayresults)

    for epoch_num in range(EPOCHS):
        print("epoch: {} of {}".format(epoch_num, EPOCHS))
        for step_num, item in enumerate(train_loader):
            optimizer.zero_grad()

            data = item[0]
            target = item[1].double()
            pred = net(data)

            loss = calc_loss(pred, target)

            plt_y = loss.detach()
            plt_x = step_num * (epoch_num + 1)
            plt.scatter(plt_x, plt_y, c='b', s=0.1)

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
    allbets = []
    for test_step, test_item in enumerate(test_loader):

            print(test_step)
            print(test_item)
            print('ethan just has a horrible sense of what is "chill" to wear')
            test_data = test_item[0]
            target = test_item[1].double()
            with torch.no_grad():

                pred = net(test_data)
                probs.append(pred)
                test_loss = calc_loss(pred, target)

                if test_step % 10 == 1:
                    print('step: {}'.format(step_num))
                    print('input: {}'.format(test_data))
                    print('pred: {}'.format(pred[0]))
                    print('target: {}'.format(target[0]))

                if abs(test_loss) < p_val:
                    correct += 1     

                h_line = test_item[0][5]
                home_result = target
                h_ev = pred * h.calc._eq(h_line) - (1 - pred)
                if home_result == 1:
                    roi_home =h.calc._eq(h_line)

                if home_result == 0:
                    roi_home = -1

                if h_ev > 0:
                    h_bets = [pred, h_line, h_ev, roi_home, home_result]

                    allbets.append(h_bets)

    all_df = pd.DataFrame(allbets, columns = ['winprob', 'line', 'ev', 'roi', 'winner'])
    print(all_df['roi'].sum())
    print(all_df)
    print('blebber catch')

    print('correct guesses: {} / total guesses: {}'.format(correct, test_step))                

    length = len(home_mls)
    n = 0
    right = 0
    wrong = 0
    h_bets = []
    a_bets = []
    allbets = []


    for i in range(length):
        h_line = home_mls[i]
        a_line = away_mls[i]
        #print(probs[i])
        h_winprob = probs[i]
        a_winprob = 1 - h_winprob
        home_result = homeresults[i]
        
        h_ev = h_winprob * h.calc._eq(h_line) - a_winprob 
        a_ev = a_winprob * h.calc._eq(a_line) - h_winprob

        h_ev.tolist()
        a_ev.tolist()

        if home_result == 1:
            roi_home =h.calc._eq(h_line)
            roi_away = -1

        if home_result == 0:
            roi_home = -1
            roi_away = h.calc._eq(a_line)

        if a_ev > 0:
            a_bets = [a_winprob, a_line, a_ev, roi_away, home_result]
            a_bets.append(a_bets)
            allbets.append(a_bets)

        if h_ev > 0:
            h_bets = [h_winprob, h_line, h_ev, roi_home, home_result]
            h_bets.append(h_bets)
            allbets.append(h_bets)

        if h_winprob > .5 and home_result == 1:
            right +=1
        if a_winprob > .5 and home_result == 0:
            right +=1

        if a_winprob < .5 and home_result == 0:
            wrong +=1

        if a_winprob > .5 and home_result == 1:
            wrong +=1

        print(h_line)
        print(h_winprob)
        print(a_line)

        plt.scatter(i, n, c='r', s=0.1)

    all_df = pd.DataFrame(allbets, columns = ['winprob', 'line', 'ev', 'roi', 'winner'])
    print('all_df'.format(all_df))
    
    total_roi = all_df['roi'].sum() 
    
    print(total_roi)

    print(right / (wrong+right))

plt.show()

