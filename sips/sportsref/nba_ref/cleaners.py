import os

import pandas as pd

GAMES_DATA = '/home/sippycups/absa/sips/data/nba/games/'

LINE_SCORES = ['win', 'team', 'q_1', 'q_2', 'q_3', 'q_4', 'T']

FOUR_FACTORS = ['win', 'team', 'Pace',
                'eFG%', 'TOV%', 'ORB%', 'FT/FGA', 'ORtg']

BASIC = ['index', 'Starters',       'MP',       'FG',      'FGA',      'FG%',
         '3P',      '3PA',      '3P%',       'FT',      'FTA',      'FT%',
         'ORB',      'DRB',      'TRB',      'AST',      'STL',      'BLK',
         'TOV',       'PF',      'PTS',      '+/-']

ADVANCED = ['index', 'Starters', 'MP', 'TS%', 'eFG%', '3PAr', 'FTr', 'ORB%',
            'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'ORtg', 'DRtg']
COL_INDEX = {
    'line_score': LINE_SCORES,
    'four_factors': FOUR_FACTORS,
    'basic': BASIC,
    'advanced': ADVANCED,
}  # drop rename


def clean_dir(dir=GAMES_DATA, write=False):
    fns = os.listdir(dir)
    clean_given_fns(fns, write=write)


def clean_given_fns(fns: list, write=False):

    table_types = list(COL_INDEX.keys())
    fns.remove('index.csv')
    for i, fn in enumerate(fns):
        full_fn = GAMES_DATA + fn
        print(f'{i}: {fn}')
        if 'line_score' in fn:
            df = drop_rename_from_fn(full_fn, COL_INDEX['line_score'])
        elif 'four_factors' in fn:
            df = drop_rename_from_fn(full_fn, COL_INDEX['four_factors'])
        elif 'basic' in fn:
            df = drop_rename_from_fn(full_fn, COL_INDEX['basic'])
        elif 'advanced' in fn:
            df = drop_rename_from_fn(full_fn, COL_INDEX['advanced'])
        elif 'shooting' in fn:
            # drop first col
            df = drop_ith_col(full_fn, 0)
        else:
            continue

        print(df)
        if write:
            df.to_csv(full_fn)
            # break
        if i == 10:
            return


def drop_rename_from_fn(fn, cols):
    print(fn)
    df = pd.read_csv(fn)
    df = drop_rename(df, cols)
    game_id = full_fn_to_game_id(fn)
    df['game_id'] = game_id
    return df


def drop_rename(df, columns):
    df = df.drop(0)
    df.columns = columns
    if 'index' in df.columns:
        df = df.drop('index', axis=1)
    return df


def drop_ith_col(fn, i):
    df = pd.read_csv(fn)
    df = df.drop(df.columns[i], axis=1)
    return df


def key_to_game_id(s: str):
    return s.split('_')[0]

def full_fn_to_game_id(s:str):
    s = key_to_game_id(s)
    return s.split('/')[-1]

if __name__ == "__main__":
    clean_dir()
