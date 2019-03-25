import gym
import random
import numpy as np

import helpers as h
import gym_sip

import torch
import torch.nn as nn
import torch.nn.functional as F

import lines as ll

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   
        
        self.fc2 = nn.Linear(50, 25)
        self.fc2.weight.data.normal_(0, 0.1)
        
        self.fc3 = nn.Linear(25, 10)
        self.fc3.weight.data.normal_(0, 0.1)   
        
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
        print('this is prev pred: {} more stuff'.format(51))

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
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
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        print('q_eval: {}'.format(q_eval))
        print('q_next: {}'.format(q_next))
        print('q_target: {}'.format(q_target))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
	EPOCHS = 50000

	# main init
	sip = ll.Sippy(self.fn, header, self.gt)
	env = gym.make('Sip-v0').unwrapped
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	BATCH_SIZE = 200
	LR = 0.01                   # learning rate
	EPSILON = 0.9               # greedy policy
	GAMMA = 0.99                 # reward discount
	TARGET_REPLACE_ITER = 100   # target update frequency
	MEMORY_CAPACITY = 2000

	N_ACTIONS = env.action_space.n
	N_STATES = env.observation_space.shape[0]
	ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

	steps_done = 0
	reward_sum = 0
	reward_list = []

	dqn = DQN()


	while True:
		
		self.sip.step()	

		for game in self.sip.games():
			game.__repr__()
	        cur_state = game.return_row()

	        a = dqn.choose_action(cur_state)  # give deep q network state and return action
	        next_state, r, d, odds = env.step(a)  # next state, reward, done, odds

	        dqn.store_transition(cur_state, a, r, next_state)

	        print('step: {}'.format(dqn.memory_counter))
	        print('game_id: {}'.format(env.game.id))
	        print('state: {}'.format(cur_state))
	        print('reward: {}'.format(r))
	        print('env.money: {}'.format(env.money))
	        print(odds)
	        print('\n')



	        if dqn.memory_counter > MEMORY_CAPACITY:
	            dqn.learn()

	        if not d:
	        	print('.')
	        else:
	            break

