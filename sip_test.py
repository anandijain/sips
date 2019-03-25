import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import random
import numpy as np

import helpers as h
import gym_sip


if __name__ == "__main__":
    EPOCHS = 50000

    # main_init()
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

    class Net(nn.Module):
        def __init__(self, ):
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


    steps_done = 0
    reward_sum = 0
    reward_list = []

    dqn = DQN()

    # try:
    #     dqn.eval_net.load_state_dict(torch.load('models/eval.ckpt'))
    #     dqn.target_net.load_state_dict(torch.load('models/target.ckpt'))
    # except FileNotFoundError:
    #     pass

    num_games = 50

    x_axis = []
    y_axis = [] 
    homesales = [] 
    awaysales = [] 
    money_list = [] 
    place_in_game_axis = []

    num_profitable_steps = 0
    num_unprofitable_steps = 0

    for game_num in range(num_games):  # run on set number of games

        if game_num % (num_games / 10) == 0:
            print("GAME: ", end='')
            print(game_num)
            print('num profit steps: {}'.format(num_profitable_steps))
            print('num unprofit steps: {}'.format(num_unprofitable_steps))
            print('\n')

        cur_state, d = env.next()

        i = 0
        while True:
            a = dqn.choose_action(cur_state)  # give deep q network state and return action
            next_state, r, d, odds = env.step(a)  # next state, reward, done, odds

            dqn.store_transition(cur_state, a, r, next_state)

            if r > 0:
                num_profitable_steps += 1
            elif r < 0:
                num_unprofitable_steps += 1

            print('step: {}'.format(dqn.memory_counter))
            print('game_id: {}'.format(env.game.id))
            print('reward: {}'.format(r))
            print('env.money: {}'.format(env.money))
            print(odds)
            print('\n')

            i += 1

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()

            if not d:
                if env.init_a_bet.a_odds != 0 and env.init_a_bet.h_odds != 0:

                    awaysale_price = h.net_given_odds(env.init_a_bet, odds)
                    homesale_price = h.net_given_odds(env.init_h_bet, odds)

                    points_sum = env.cur_state[3] + env.cur_state[4]
                    x = game_num + i/env.game.game_len

                    homesales.append(homesale_price)
                    awaysales.append(awaysale_price)
                    money_list.append(env.money)

                    x_axis.append(game_num + i/env.game.game_len)
                    place_in_game_axis.append(i)
                    reward_list.append(r)

                    if homesale_price > 1000: 
                        print("homesale_price high")
                        print(homesale_price)
                        print(awaysale_price)
                        env.init_h_bet.__repr__()
                        print(odds)
            else:
                break

# torch.save(dqn.eval_net.state_dict(), 'models/eval.ckpt')
# torch.save(dqn.target_net.state_dict(), 'models/target.ckpt')

# print(env.money)
# print(len(x_axis))
# np_x_axis = np.array(x_axis)
# np_reward_list = np.array(reward_list)

# np_homesales = np.array(homesales)
# np_awaysales = np.array(awaysales)
# np_money_list = np.array(money_list)

# # np_rl = np.array(reward_list)
# # np_rl = np_rl.astype(float)

# plt.scatter(np_x_axis, np_homesales, c='blue', s=1, alpha=0.3)
# plt.scatter(np_x_axis, np_awaysales, c='green', s=1, alpha=0.3)
# plt.scatter(np_x_axis, np_money_list, c='yellow', s=2.5, alpha=0.5)
# plt.scatter(np_x_axis, np_reward_list, c='red', s=2.5, alpha=0.5)

# # plt.plot(np_x_axis, np_reward_list)

# plt.show()
