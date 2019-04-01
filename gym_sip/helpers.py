# helper functions for Sip OpenAI Gym environment
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class Df(Dataset):
    # predicting next line in time
    def __init__(self, np_df, unscaled):
        self.data_len = len(np_df)
        self.data = np_df
        self.unscaled_data = unscaled
        print(self.data_len)

    def __getitem__(self, index):
        # line = self.data.iloc[index]
        line = self.data[index]
        line_tensor = torch.tensor(line)
        unscaled_line = self.unscaled_data[index]
        unscaled_tensor = torch.tensor(unscaled_line)        
        return line_tensor, unscaled_tensor

    def __len__(self):
        return self.data_len


class DfGame(Dataset):
    # given dictionary of padded games
    def __init__(self, games):
        self.games_dict = games
        self.games = list(self.games_dict.values())
        self.ids = list(self.games_dict.keys())
        self.game_id = self.ids[0]
        self.game = self.games[0] 
        self.data_len = len(self.games)  # not padding aware because shape is constant
        self.game_len = len(self.games_dict[self.game_id])  # assuming all games same length

    def __getitem__(self, index):
        self.game = self.games[index]
        first_half, second_half = split_game_in_half(self.game)
        return self.game

    def __len__(self):
        return self.data_len


class DfPastGames(Dataset):
    # each line of csv is a game, takes in pandas and a list of strings of which columns are labels
    def __init__(self, df, train_columns=['a_pts', 'h_pts']):
        self.df = df
        self.train_columns = train_columns
        self.data_len = len(self.df)

        self.labels = self.df[self.train_columns].values
        # self.labels= sk_scale(self.labels)
        self.labels = torch.tensor(self.labels)
        self.labels_shape = len(self.labels)

        self.data = self.df.drop(self.train_columns, axis=1).values
        # self.data = sk_scale(self.data)
        self.data = torch.tensor(self.data)
        self.data_shape = len(self.data[0])

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data_len        


headers = ['a_team', 'h_team', 'sport', 'league', 'game_id', 'cur_time',
           'a_pts', 'h_pts', 'secs', 'status', 'a_win', 'h_win', 'last_mod_to_start', 'last_mod_lines'
           'num_markets', 'a_odds_ml', 'h_odds_ml', 'a_hcap_tot', 'h_hcap_tot', 'game_start_time']


def get_games(fn='../data/nba2.csv'):
    # takes in fn and returns python dict of pd dfs 
    df = get_df(fn)
    games = chunk(df, 'game_id')
    games = remove_missed_wins(games)
    return games


def get_df(fn='../data/nba2.csv'):
    raw = csv(fn)
    df = one_hots(raw, ['league', 'a_team', 'h_team'])
    # df = one_hots(raw, ['a_team', 'h_team', 'w_l'])
    return df


def chunk(df, col='game_id'):
    # returns a python dict of pandas dfs, splitting the df arg by unique col value
    # df type pd df, col type string
    games = {key: val for key, val in df.groupby(col)}
    return games


def csv(fn='../data/nba2.csv'):
    # takes in file name string, returns pandas dataframe
    print(fn)
    df = pd.read_csv(fn)
    df = drop_null_times(df)
    df = df.drop(['sport', 'lms_date', 'lms_time'], axis=1)
    # df = df.drop(['date'], axis=1)
    return df.copy()


def one_hots(df, cols):
    # df is pandas df, cols is string list
    one_hot_df = pd.get_dummies(data=df, columns=cols, sparse=False)
    return one_hot_df


def remove_missed_wins(games):
    # takes in a dictionary or list of df games
    games_len = len(games)
    if isinstance(games, dict):
        for g_id in list(games.keys()): 
            if len(games[g_id]['a_win'].unique()) + len(games[g_id]['h_win'].unique()) != 3:
                del games[g_id]

    elif isinstance(games, list):
        for elt in games:
            if len(elt['a_win'].unique()) + len(elt['h_win'].unique()) != 3:
                games.remove(elt)
    else:
        raise TypeError('games argument must be dict or list')

    # if games_len != len(games):
    #     print('before: {}'.format(games_len))
    #     print('after: {}'.format(len(games)))

    return games


def drop_null_times(df, columns=['lms_date', 'lms_time']):
    # given pandas df and list of strings for columns. convert '0' values to np.datetime64
    init_len = len(df)
    
    print('dropping null times from columns: {}'.format(columns))
    print('df init length: {}'.format(init_len))
    
    for col in columns:
        df[col] = df[col].replace('0', np.nan)
        # df[col] = pd.to_datetime(df[col])  # TODO

    df = df.dropna()

    after_len = len(df)
    delta = init_len - after_len

    print('df after length: {}'.format(after_len))
    print('delta (lines removed): {}'.format(delta))
    return df


def split_game_in_half(game):
    game_len = len(game)
    first_half = game[:game_len//2, ]
    second_half = game[game_len//2:, ]
    return first_half, second_half


def apply_min_game_len(games, min_lines=500):
    # given dict of game dataframes and an integer > 0 for the minimum length of a game in csv lines 
    print('applying minimum game len of : {}'.format(min_lines))
    print('before apply: {}'.format(len(games)))
    for key, value in games.copy().items():
        game_len = len(value)
        if game_len < min_lines:
            print('deleted game_id: {}'.format(key))
            print('had length: {}'.format(game_len))
            del games[key]
    print('after apply: {}'.format(len(games)))
    return games


def df_info(df):
    # TODO
    # given pd df, return the general important info in console
    # num games, teams, etc 
    pass


def random_game(games):
    game_id, game = random.choice(list(games.items()))
    print('game_id: {}'.format(game_id))
    return gam


def label_split(df, col):
    # give column to be predicted given all others in csv
    # df is pd, col is string
    Y = df[col]
    X = df.drop(col, axis=1)
    return X, Y


def train_test(df, train_pct=0.5):
    test = df.sample(frac=train_pct, random_state=None)
    train = df.drop(test.index)
    return train.copy(), test.copy()



def sk_scale(data):
    # vals = data.to_numpy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    return scaled


def select_dtypes(df, dtypes=['number']):
    return df.select_dtypes(include=dtypes)


def teams_given_state(state):  
    # given np array, representing a state (csv_line). returns tuple of teams
    return state


def games_to_padded_tensors(games, length=1200):
    # takes in games dict and returns dict of padded length tensors
    copy = games.copy()
    for game in copy:
        game_df = games[game]
        game_np = game_df.values
        games[game] = pad_tensor(game_np)
    return games


def get_t_games(df):
    games = chunk(df)
    games = remove_missed_wins(games)
    t_games = games_to_padded_tensors(games)
    return t_games


def pad_tensor(game, length=1200):
    tensor = torch.from_numpy(game)
    return torch.cat([tensor, tensor.new(length - tensor.size(0), * tensor.size()[1:]).zero_()])


def dates(df):
    # convert ['lms_date', 'lms_time'] into datetimes
    df['datetime'] = df['lms_date'] + ' ' + df['lms_time']
    df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True, errors='coerce')
    df['datetime'] = pd.to_numeric(df['datetime'])
    df = df.drop(['lms_date', 'lms_time'], axis=1)
    # df = df.drop(df['datetime'], axis=1)
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


def eq_to_odd(equity):
    if equity > 1:
        odd = 100 * equity
        return odd
    elif equity <= 1:
        odd = -100/equity
        return odd
    elif equity == 0:
        return 0


def act(a):
    # simple function to easily change the action number into a string
    # returns string
    if a == 0:
        return 'BOUGHT AWAY'
    elif a == 1:
        return 'BOUGHT HOME'
    elif a == 2:
        return 'SKIP'
    else: 
        return 'action outside of defined actions'


def net(bet, bet2):
    # given a bet pair (bet + hedge)
    # input: Hedge class, output float
    # env.is_valid() should have already caught zero odds lines
    # a full hedge equates the profit, so
    # bet.amt * _eq(bet.a) should be equal to bet2.amt * _eq(bet2.h)
    bet_sum = bet.amt + bet2.amt
    if bet.team == 0:
        return bet.amt * _eq(bet.a_odds) - bet2.amt
    else:
        return bet.amt * _eq(bet.h_odds) - bet2.amt


def bet_amt(money):
    # return 0.05 * money + 100  # 100 is arbitrary
    return 100


def hedge_amt(bet, cur_odds):
    # takes in Bet 1 and calculates the 
    if bet.team == 0:
        return (bet.amt * (_eq(bet.a_odds) + 1))/ (_eq(cur_odds[1]) + 1)
    else:
        return (bet.amt * (_eq(bet.h_odds) + 1)) / (_eq(cur_odds[0]) + 1)


def net_given_odds(bet, cur_odds):
    bet2_amt = hedge_amt(bet, cur_odds)
    bet_sum = bet.amt + bet2_amt
    if bet.team == 0:
        return bet.amt * _eq(bet.a_odds) - bet2_amt
    else:
        return bet.amt * _eq(bet.h_odds) - bet2_amt
