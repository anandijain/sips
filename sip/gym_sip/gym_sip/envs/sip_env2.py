import gym
import h
import random
import numpy as np



ACTION_BUY_A = 0
ACTION_BUY_H = 1
ACTION_SKIP = 2


class BetaState:
    def __init__(self, game):
        self.game = game
        self.game_len = len(game)  # used often
        self.index = 0
        self.id = game[0: 0]
        self.cur_state = self.game.iloc[self.index]
        # print("imported {}".format(self.id))

    def next(self):
        if self.game_over():
            return self.cur_state, True
        self.cur_state = self.game.iloc[self.index, 0:]
        self.index += 1
        print('index: {}'.format(self.index))
        return self.cur_state, False

    def reset(self):
        self.index = 0

    def shape(self):
        return self.game.shape

    def a_pts(self):
        return float(self.game.iloc[self.index, 2])
        # return int(self.game_a_odds[self.index])

    def h_pts(self):
        return float(self.game.iloc[self.index, 3])
        # return int(self.game_h_odds[self.index])

    def game_over(self):
        csv_end = self.index > (self.game_len - 1)
        return csv_end 


class SipEnv2(gym.Env):

    def __init__(self, fn='../../data/bangout3.csv'):
        self.games = h.get_games(fn)
        self.game = None
        self.game_id = 0
        self.game_ids = list(self.games.keys())
        self.game_counter = 0
        self.new_game()
        self.last_pick = None
        self.cur_state = self.game.cur_state
        self.action = None
        self.combos = []
        self.game_combos = 0
        self._scores()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-1e7, high=1e7, shape=(len(self.cur_state), 0))


    def step(self, action):  # action given to us from test.py
        self.action = action   
        reward = 0 
        self.cur_state, done = self.game.next()  # goes to the next timestep in current game
        self._scores()
        print('scores {}'.format(self.scores))

        if done is True: 
            if self.last_pick is None:
                return self.cur_state, 0, True, self.scores
            else:
                reward = self.forgot_to_combo()
                return self.cur_state, reward, done, self.scores

        if self.last_pick is not None:
            self.last_pick.wait_amt += 1
            
        reward = self.act()  # MAIN ACTION CALL

        if reward == None:
            return self.cur_state, 0, True, self.scores
        # place_in_game = self.game.index / self.game.game_len
        # if reward > 0:
        #     reward = reward / place_in_game

        return self.cur_state, reward, done, self.scores

    def act(self):
        if self.action == ACTION_SKIP:
            print(self.last_pick)
            return 0 

        elif self.last_pick is None:  # if last bet != None, then this bet is a hedge
            self._pick()  
            return 0  # reward for getting equity?
        elif self.last_pick.team == self.action:  # betting on same team twice
            return 0
        else:
            net = self._combo()
            return net

    def _pick(self):
        # we don't update self.money because we don't want it to get a negative reward on _bet()
        # print("bet*")
        self.last_pick = Pick(self.action, self.scores, self.cur_state)
        print('helloworld')

    def _combo(self):
        combo_pick  = Pick(self.action, self.scores, self.cur_state)

        net = h.net_score(self.last_pick, combo_pick)
        combo = Combo(self.last_pick, combo_pick)
        # hedge.__repr__()

        self.combos.append(combo)
        self.last_pick = None

        self.game_combos += 1
        return combo.net

    def new_game(self):
        self.game_combos = 0
        self.last_pick = None  # once a game has ended, bets are cleared

        self.game_id = self.game_ids[self.game_counter]
        self.game = BetaState(self.games[self.game_id])

        # self.get_teams_from_state()

        # print(self.game.cur_state)
        self.game_counter += 1


    def forgot_to_combo(self):
        # print('forgot to hedge')
        reward = -1000
        self.last_pick.__repr__()
        # print(self.last_pick.wait_amt)
        return reward

    def _scores(self):
        self.scores = (self.cur_state[2], self.cur_state[3])

    def next(self):
        self.new_game()
        self.cur_state, done = self.game.next()
        return self.cur_state, done



class Pick:
# class storing bet info, will be stored in pair (hedged-bet)
# might want to add time into game so we can easily aggregate when it is betting in the game
# possibly using line numbers where they update -(1/(x-5)). x=5 is end of game
# maybe bets should be stored as a step (csv line) and the bet amt and index into game.
    def __init__(self, action, scores, cur_state):

        self.team = action  # 0 for away, 1 for home
        self.a_pts = scores[0]
        self.h_pts = scores[1]
        self.cur_state = cur_state
        self.wait_amt = 0


    # def __repr__(self):
        # simple console log of a bet
        # print(h.act(self.team))
        # print(' | team: ' + str(self.team))
        # print('a_odds: {}'.format(scores))


class Combo:
    def __init__(self, pick, pick2):
        # input args is two Bets
        self.net = h.net_score(pick, pick2)
        self.pick = pick
        self.pick2 = pick2
      #  print(self.net + "LALA")

    # def __repr__(self):
        # print("[BET 1 of 2")
        # self.pick.__repr__()
        # print("BET 2 of 2")
        # self.pick2.__repr__()
        # print('advancment: {}'.format(self.net))
        # print('steps waited: {}' + str(self.pick.wait_amt))

        # print('\n')
