import gym
import gym_sip

import random
import numpy as np

import helpers as h
import sipqn

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 


env = gym.make('Sip-v1').unwrapped



if __name__ == "__main__":
    EPOCHS = 50000

    # main_init()
    env = gym.make('Sip-v1').unwrapped
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

    dqn = sipqn.DQN(N_STATES,N_ACTIONS, ENV_A_SHAPE)


    num_games = 50
    


    num_profitable_steps = 0
    num_unprofitable_steps = 0

    for game_num in range(num_games):  # run on set number of games

        if game_num % (num_games / 10) == 0:
        	cur_state, d = env.next()

        i = 0
        while True:
            a = dqn.choose_action(cur_state)  # give deep q network state and return action
            next_state, r, d, odds = env.step(a)  # next state, reward, done, odds
            print('action: {}'.format(a))
            print('reward: {}'.format(r))
            dqn.store_transition(cur_state, a, r, next_state)

            if r > 0:
                num_profitable_steps += 1
            elif r < 0:
                num_unprofitable_steps += 1

            i += 1

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()

            if d:
                break


torch.save(dqn.state_dict(), './models/first')

