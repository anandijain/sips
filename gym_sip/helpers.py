# helper functions for Sip OpenAI Gym environment
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
import math

from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence

teams = ['Atlanta Hawks', 'Boston Celtics',
       'Brooklyn Nets', 'Charlotte Hornets',
       'Chicago Bulls', 'Cleveland Cavaliers',
       'Dallas Mavericks', 'Denver Nuggets',
       'Detroit Pistons', 'Golden State Warriors',
       'Houston Rockets', 'Indiana Pacers',
       'Los Angeles Clippers', 'Los Angeles Lakers',
       'Memphis Grizzlies', 'Miami Heat',
       'Milwaukee Bucks', 'Minnesota Timberwolves',
       'New Orleans Pelicans', 'New York Knicks',
       'Oklahoma City Thunder', 'Orlando Magic',
       'Philadelphia 76ers', 'Phoenix Suns',
       'Portland Trail Blazers', 'Sacramento Kings',
       'San Antonio Spurs', 'Toronto Raptors',
       'Utah Jazz', 'Washington Wizards']


class Df(Dataset):
    def __init__(self, df, prev_n=5, next_n=1):
        df = df.astype(float)
        self.df = df.values
        self.tdf = torch.from_numpy(sk_scale(self.df))
        
        self.total_len = len(self.tdf)
        self.num_cols = len(self.df[0])

        self.prev_n = prev_n
        self.next_n = next_n

        self.past = None
        self.future = None
        
        item = self.__getitem__(500)

        self.shape = item[0].shape
        self.out_shape = item[1].shape
        
    def __getitem__(self, index):
        self.past = self.tdf[index - self.prev_n:index]
        # if len(self.past) < prev_n:

        if index < self.total_len - self.next_n - 1:
            self.past = self.tdf[index - self.prev_n : index] 
            # self.past = torch.from_numpy(self.past)

            self.future = self.tdf[index: index + self.next_n]
            # self.future_n = torch.from_numpy(self.future_n)

            self.past = torch.cat([self.past, self.past.new(self.prev_n - self.past.size(0), * self.past.size()[1:]).zero_()])
            self.future = torch.cat([self.future, self.future.new(self.next_n - self.future.size(0), * self.future.size()[1:]).zero_()])
            return (self.past, self.future)

        else:
            return None

    def __len__(self):
        return self.total_len


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
        print(self.data)
        self.data = torch.tensor(self.data)
        self.data_shape = len(self.data[0])

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data_len        


class DfCols(Dataset):
    # each line of csv is a game, takes in pandas and a list of strings of which columns are labels
    def __init__(self, df, train_cols=['quarter', 'secs'], label_cols=['a_pts', 'h_pts']):
        self.df = df.sort_values(by='cur_time')
        self.data_len = len(self.df)
        
        self.train_cols = train_cols
        self.label_cols = label_cols

        self.labels = self.df[self.label_cols]
        self.labels = self.labels.astype(float).values
        self.labels = sk_scale(self.labels)
        self.labels = torch.tensor(self.labels)

        self.labels_shape = len(self.labels)
        if self.train_cols is None:
            self.data = self.df.drop(self.label_cols, axis=1)
        else:
            self.data = self.df[self.train_cols]
        self.data = self.data.astype(float).values
        self.data = sk_scale(self.data)
        self.data = torch.tensor(self.data)

        self.data_shape = len(self.data[0])

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data_len


headers = ['a_team', 'h_team', 'sport', 'league', 'game_id', 'cur_time',
           'a_pts', 'h_pts', 'secs', 'status', 'a_win', 'h_win', 'last_mod_to_start', 'last_mod_lines',
           'num_markets', 'a_ml', 'h_ml', 'a_hcap_tot', 'h_hcap_tot', 'game_start_time']


def df_cols():
    df = get_df()
    try:
        loader = DfCols(df)
    except Exception:
        loader = None
        pass
    return df, loader


def get_games(fn='./data/nba2.csv'):
    # takes in fn and returns python dict of pd dfs 
    df = get_df(fn)
    games = chunk(df)
    games = remove_missed_wins(games)
    return games


def get_df(fn='./data/nba2.csv'):
    raw = csv(fn)
    df = one_hots(raw, ['league', 'a_team', 'h_team'])
    df = dates(df)
    # df = one_hots(raw, ['a_team', 'h_team', 'w_l'])
    return df


def chunk(df, col='game_id'):
    # returns a python dict of pandas dfs, splitting the df arg by unique col value
    # df type pd df, col type string
    games = {key: val for key, val in df.groupby(col)}
    return games


def csv(fn='./data/nba2.csv'):
    # takes in file name string, returns pandas dataframe
    print(fn)
    df = pd.read_csv(fn)
    df = drop_null_times(df)
    df = df.drop('sport', axis=1)
    # df = df.drop(['date'], axis=1)
    return df


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

    if games_len != len(games):
        print('before: {}'.format(games_len))
        print('after: {}'.format(len(games)))

    return games


def drop_null_times(df, columns=['lms_date', 'lms_time']):
    # given pandas df and list of strings for columns. convert '0' values to np.datetime64
    init_len = len(df)
    
    for col in columns:
        df[col] = df[col].replace('0', np.nan)

    df = df.dropna()

    after_len = len(df)
    delta = init_len - after_len

    print('len(df) before: {}, after length: {}, delta: {}'.format(init_len, after_len, delta))
    return df


def dates(df):
    # convert ['lms_date', 'lms_time'] into datetimes
    df['datetime'] = df['lms_date'] + ' ' + df['lms_time']
    # print(df['datetime'])
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    
    dt = df['datetime'].dt
    df['time_vals'] = pd.to_numeric(df['datetime'])
    date_categories = [dt.year, dt.month, dt.week, dt.day, 
                        dt.hour, dt.minute, dt.second, dt.dayofweek, dt.dayofyear]
    col_names = ['year', 'month', 'week', 'day', 'hour', 'minute', 'second', 'dayofweek', 'dayofyear']

    for i in range(len(date_categories) - 1):
        df[col_names[i]] = date_categories[i]

    df = df.drop(['lms_date', 'lms_time', 'datetime'], axis=1)
  
    # print('df after h.dates: {}'.format(df))
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
    # takes in pandas or np array and returns corresponding type
    if isinstance(df, pd.core.frame.DataFrame):  # pandas
        train = df.sample(frac=train_pct, random_state=None)
        test = df.drop(train.index)
        return train, test
    elif isinstance(df, np.ndarray): # np
        length = len(df)
        split_index = round(length * train_pct) 
        train = df[:split_index, :]
        test = df[split_index:, :]
        return train, test
    else:
        raise TypeError("please provide a numpy array or a pandas dataframe")



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


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


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

def net_score(pick, pick2):
    if pick.team == 0:
        init_diff = pick.a_pts - pick.h_pts
        final_diff = pick2.a_pts - pick2.h_pts
    elif pick.team == 1:
        init_diff = pick.h_pts - pick.a_pts
        final_diff = pick2.h_pts - pick2.a_pts

    return final_diff - init_diff

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


def onehot_teams(a_team, h_team):  #take in 2 teams playing and onehot to consistent format
       one_hotted = []
       for cur_team in [a_team, h_team]:

              for index, elt in enumerate(teams):
                     if cur_team == elt:
                            onehot1 = pd.Series([0 for i in range(index)])
                            onehotreal = pd.Series([1])
                            onehot2 = pd.Series([0 for i in range(30-index-1)])
                            y = onehot1.append(onehotreal)
                            final = y.append(onehot2)
                            final = final.reset_index(drop=True)
                            one_hotted.append(final)
       return one_hotted


def series_mean(series):
    list = series.tolist()

    sum = 0 
    for elt in list:
        sum += elt

    mean = sum / len(list)

    return mean

def series_std_dev(series, mean):
    list = series.tolist()

    var = 0
    for elt in list:
        var += (elt - mean)**2

    std_dev = math.sqrt(var / len(list))

    return std_dev


def df_means(df):
    list_of_mean = []
    for col in df:
        mean = series_mean(df[col])
        list_of_mean.append(mean)

    return list_of_mean

def df_std_devs(df, df_means):
    list_of_std_dev = []
    for index, col in enumerate(df):
        list_of_std_dev.append(series_std_dev(df[col], df_means[index]))

    return list_of_std_dev


def df_normalize(df):
    means = df_means(df)
    devs = df_std_devs(df, means)
    normed_list = []
    for index, col in enumerate(df):
        for elt in df[col]:
            try:
                normed_list.append((elt - means[index] / devs[index]))
            except ZeroDivisionError:
                continue

    return normed_list




