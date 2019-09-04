import gym
import gym_sip

import sipqn

import random
import numpy as np
from h import helpers as h

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

if __name__ == "__main__":
    EPOCHS = 1

    # main_init()
    env = gym.make('Sip-v0').unwrapped
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 200
    LR = 0.01                   # learning rate
    EPSILON = 0.9               # greedy policy
    GAMMA = 0.99                 # reward discount
    TARGET_REPLACE_ITER = 100   # target update frequency
    MEMORY_CAPACITY = 2000

    N_STATES = env.observation_space.shape[0]
    N_ACTIONS = env.action_space.n
    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


    steps_done = 0
    reward_sum = 0
    reward_list = []

    dqn = sipqn.DQN(N_STATES, N_ACTIONS, ENV_A_SHAPE)

    # try:
    #     dqn.eval_net.load_state_dict(torch.load('./models/eval.ckpt'))
    #     dqn.target_net.load_state_dict(torch.load('./models/target.ckpt'))
    # except FileNotFoundError:
    #     pass

    num_games = 100

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
        print(cur_state)

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

torch.save(dqn.eval_net.state_dict(), './models/eval.ckpt')
torch.save(dqn.target_net.state_dict(), './models/target.ckpt')

print(env.money)
print(len(x_axis))
np_x_axis = np.array(x_axis)
np_reward_list = np.array(reward_list)

np_homesales = np.array(homesales)
np_awaysales = np.array(awaysales)
np_money_list = np.array(money_list)

# np_rl = np.array(reward_list)
# np_rl = np_rl.astype(float)

plt.scatter(np_x_axis, np_homesales, c='blue', s=1, alpha=0.3)
plt.scatter(np_x_axis, np_awaysales, c='green', s=1, alpha=0.3)
plt.scatter(np_x_axis, np_money_list, c='yellow', s=2.5, alpha=0.5)
plt.scatter(np_x_axis, np_reward_list, c='red', s=2.5, alpha=0.5)

# plt.plot(np_x_axis, np_reward_list)

plt.show()
