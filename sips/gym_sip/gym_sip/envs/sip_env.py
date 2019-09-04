import gym
import random
import h
import numpy as np

from gym import spaces

# Macros for actions
ACTION_BUY_A = 0
ACTION_BUY_H = 1
ACTION_SKIP = 2

# Starting bank
AUM = 10000


class SippyState:
    # SippyState is a Gym state
    # Using pd df with unique 'game_id'

    def __init__(self, game):
        self.game = game  # store in State for repeated access
        self.game_len = len(game)  # used often
        self.id = self.ids()[0]
        self.index = 0
        self.cur_state = self.game.iloc[self.index]
        self.teams = h.teams_given_state(self.cur_state)
        self.team_won = False
        print("imported {}".format(self.id))

    def next(self):
        if self.game_over():
            return self.cur_state, True
        self.cur_state = self.game.iloc[self.index, 0:]
        self.index += 1
        return self.cur_state, False

    def reset(self):
        self.index = 0

    def shape(self):
        return self.game.shape

    def a_odds(self):
        return int(self.game.iloc[self.index, 14])
        # return int(self.game_a_odds[self.index])

    def h_odds(self):
        return int(self.game.iloc[self.index, 15])
        # return int(self.game_h_odds[self.index])

    def game_over(self):
        csv_end = self.index > (self.game_len - 1)
        self.team_won = self.cur_state[7] == 1 or self.cur_state[8] == 1
        return csv_end or self.team_won

    def ids(self):
        ids = self.game['game_id'].unique()
        if len(ids) > 1:  # check to see if the games were not chunked correctly
            raise Exception('there was an error, chunked game has more than one id, the ids are {}'.format(ids))
        return ids


class SipEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, fn):
        self.games = h.get_games(fn, output='dict')
        self.game = None
        self.game_id = 0

        self.init_a_bet = h.Bet(100, 0, (0, 0), None)
        self.init_h_bet = h.Bet(100, 1, (0, 0), None)
        self.new_game()

        self.money = AUM
        self.last_bet = None  #
        self.cur_state = self.game.cur_state  # need to store
        self.action = None
        self.hedges = []
        self.game_hedges = 0
        self.follow_bets = 0
        self._odds()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1e7, high=1e7, shape=(len(self.cur_state), 0), dtype=np.float32)
                                                # ,dtype=np.float32)

    def step(self, action):  # action given to us from test.py
        self.action = action
        reward = 0
        self.cur_state, done = self.game.next()  # goes to the next timestep in current game
        self._odds()
        self.set_init_odds()

        if done is True:
            if self.game.team_won is False or self.last_bet is None:
                return (self.cur_state, 0, True, self.odds)
            else:
                reward = self.forgot_to_hedge()
                return (self.cur_state, reward, done, self.odds)

        if self.last_bet is not None:
            self.last_bet.wait_amt += 1

        reward = self.act()  # MAIN ACTION CALL

        if reward == None:
            return (self.cur_state, 0, True, self.odds)
        # place_in_game = self.game.index / self.game.game_len
        # if reward > 0:
        #     reward = reward / place_in_game

        return (self.cur_state, reward, done, self.odds)

    def act(self):
        if self.action == ACTION_SKIP:
            return 0
        elif self.odds[self.action] == 0:
            # can't bet on team that has zero odds
            return None
        elif self.last_bet is None:  # if last bet != None, then this bet is a hedge
            self._bet()
            return 0  # reward for getting equity?
        elif self.last_bet.team == self.action:  # betting on same team twice
            return 0
        else:
            net = self._hedge()
            self.money += net
            print(self.money)
            return net

    def _bet(self):
        # we don't update self.money because we don't want it to get a negative reward on _bet()
        amt = h.bet_amt(self.money)
        print("bet*")
        self.last_bet = h.Bet(amt, self.action, self.odds, self.cur_state)

    def _hedge(self):
        hedge_amt = h.hedge_amt(self.last_bet, self.odds)
        hedged_bet = h.Bet(hedge_amt, self.action, self.odds, self.cur_state)

        net = h.net(self.last_bet, hedged_bet)
        hedge = h.Hedge(self.last_bet, hedged_bet)
        hedge.__repr__()

        self.hedges.append(hedge)
        self.last_bet = None

        self.game_hedges += 1
        return hedge.net

    def new_game(self):
        self.game_hedges = 0
        self.follow_bets = 0

        self.init_h_bet.reset_odds()
        self.init_a_bet.reset_odds()

        self.last_bet = None  # once a game has ended, bets are cleared

        self.game_id = random.choice(list(self.games.keys()))
        self.game = SippyState(self.games[self.game_id])

        # self.get_teams_from_state()

        if self.game is None:
            del self.games[self.game_id]
            print('deleted a game')
            self.new_game()

        print(self.game.cur_state)

    def set_init_odds(self):
        if self.init_a_bet.a_odds == 0:  # check if the init a odds have been set yet
            if self.odds[0] != 0:
                print('updated init a odds: {}'.format(self.odds[0]))
                self.init_a_bet.a_odds = self.odds[0]
                self.init_h_bet.a_odds = self.odds[0]
        if self.init_a_bet.h_odds == 0:
            if self.odds[1] != 0:
                print('updated init h odds: {}'.format(self.odds[1]))
                self.init_h_bet.h_odds = self.odds[1]
                self.init_a_bet.h_odds = self.odds[1]

    def forgot_to_hedge(self):
        print('forgot to hedge')
        reward = -self.last_bet.amt
        self.last_bet.__repr__()
        print(self.last_bet.wait_amt)
        return reward

    def _odds(self):
        self.odds = (self.cur_state[12], self.cur_state[13])

    def get_state(self):
        return self.cur_state

    def next(self):
        self.new_game()
        self.cur_state, done = self.game.next()
        return (self.cur_state, done)

    def reset(self):
        self.money = AUM
        return self.next()

    def __repr__(self):
        print('index in game: ' + str(self.cur_state.index))

    def _render(self, mode='human', close=False):
        pass
