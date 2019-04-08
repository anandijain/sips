import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

BATCH_SIZE = 200
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.99                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000


class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)   
        
        self.fc2 = nn.Linear(100, 100)
        self.fc2.weight.data.normal_(0, 0.1)
        
        self.fc3 = nn.Linear(100, 100)
        self.fc3.weight.data.normal_(0, 0.1)   

        self.fc4 = nn.Linear(100, 50)
        self.fc4.weight.data.normal_(0, 0.1)   
        
        self.fc5 = nn.Linear(50, 25)
        self.fc5.weight.data.normal_(0, 0.1)   

        self.fc6 = nn.Linear(25, 25)
        self.fc6.weight.data.normal_(0, 0.1) 
        
        self.fc7 = nn.Linear(25, 10)
        self.fc7.weight.data.normal_(0, 0.1) 
        
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        actions_value = self.out(x)
        return actions_value
        # print('this is prev pred: {} more stuff'.format(51))

class DQN(object):
    def __init__(self, N_STATES, N_ACTIONS, shape):
        self.shape = shape
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                        # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.N_ACTIONS = N_ACTIONS
        self.N_STATES = N_STATES

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.shape == 0 else action.reshape(self.shape)  # return the argmax index
            print('nonrandom action')
        else:   # random
            action = np.random.randint(0, self.N_ACTIONS)
            action = action if self.shape == 0 else action.reshape(self.shape)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        # print('q_eval: {}'.format(q_eval))
        # print('q_next: {}'.format(q_next))
        # print('q_target: {}'.format(q_target))

        print('loss: {}:'.format(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
