import os
import random

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from sips.macros import macros
from sips.ml import normdf

class Scoreset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir=macros.LINES_DIR, first_n=100, last_n=20, min_len=200, sport='BASK'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.fns = [root_dir + fn for fn in os.listdir(root_dir)]
        self.first_n = first_n
        self.last_n = last_n
        self.teams = pd.read_csv('teams_updated_stats.csv')

        try:
            self.fns.remove('log.csv')
        except ValueError:
            pass 
        
        dfs = [pd.read_csv(fn) for fn in self.fns]
        self.data = []
        # print(f'len: {len(dfs)}')

        for i, df in enumerate(dfs):

            df = df[df.status != 'PRE_GAME']
            df = df[df.a_pts != 'None']
            df = df[df.h_pts != 'None']

            if df.shape[0] >= min_len:
                if df.sport.iloc[0] == sport:
                    # print(f'added {i: <2} {df.shape[0]} {df.sport.iloc[0]}')
            # print(df)
                    self.data.append(df)
            # else:
                # print(f'skipped {i: <2} {df.shape[0]} {df.sport.iloc[0]}')

        print(f'len: {len(self.data)}')

        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        df = self.data[idx]
        df_len = df.shape[0]
        y_idx = df_len - self.last_n
        a_team = df.a_team.iloc[0]
        h_team = df.h_team.iloc[0]
        a_data = self.teams[self.teams['team'] == a_team]
        h_data = self.teams[self.teams['team'] == h_team]
        
        a_data = a_data.drop('team', axis=1)
        h_data = h_data.drop('team', axis=1)

        early_points = [df.a_pts[:self.first_n],
                        df.h_pts[:self.first_n], a_data.iloc[0], h_data.iloc[0]]
        end_points = [df.a_pts[y_idx:], df.h_pts[y_idx:]]
        x = pd.concat(early_points)
        y = pd.concat(end_points)
        # print(x)
        # print(y)
        x_t = torch.tensor(x.astype(np.float32).values)
        y_t = torch.tensor(y.astype(np.float32).values)

        return {"x": x_t.view(-1), "y": y_t.view(-1)}


def col_types(df: pd.DataFrame) -> dict:
    # given a df, returns a dict where key is column name, value is dtype
    return dict(zip(list(df.columns), list(df.dtypes)))

    
def normed_scoresets(dir=macros.LINES_DIR, sport='BASK', frac=0.3):
    fns = [dir + fn for fn in os.listdir(dir)]
    str_cols=['game_id', 'h_team', 'h_team']
    dfs = [pd.read_csv(fn) for fn in fns]

    big = pd.concat(dfs)
    print(big)
    big = big[big.sport == sport]
    
    big = big[big.status != 'PRE_GAME']
    big = big[big.a_pts != 'None']
    big = big[big.h_pts != 'None']

    big = big[['game_id', 'h_team', 'a_team', 'h_pts', 'a_pts']]

    big['h_pts'] = big['h_pts'].astype(np.float32)
    big['a_pts'] = big['a_pts'].astype(np.float32)
    print(col_types(big))
    print(big)
    games = list(big.game_id.unique())
    n = len(games)

    split = int(n * frac)
    
    random.shuffle(games)
    train_ids = games[split:]
    test_ids = games[:split]

    train = big[big.game_id.isin(train_ids)]
    test = big[big.game_id.isin(test_ids)]
    print(col_types(train))
    print(col_types(test))
    normed_test = normdf.norm_testset(test, train, str_cols=str_cols)
    normed_train = normdf.to_normed(train, str_cols=str_cols)
    return normed_train.copy(), normed_test.copy()

if __name__ == "__main__":
    normed_scoresets()

