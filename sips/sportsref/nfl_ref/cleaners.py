import time
import os
import glob
import types
import re

import pandas as pd
import numpy as np

from sips.sportsref import utils
from sips.macros.sports import nba
from sips.macros import macros as m


def fix_game_info(df):
    vals = df['1'].values
    fixed = pd.DataFrame([vals], columns=df['0'])
    fixed = fixed.drop('Game Info', axis=1)
    return fixed


def transpose_fix(df):
    cols = df.iloc[:, 0].values
    # print(cols)
    ret = pd.DataFrame(df.iloc[:, 1:].T.values, columns=cols)
    # print(ret)
    ret.drop(0, inplace=True)

    # print(ret)
    return ret


def 


def get_extract():
    df = pd.read_csv(
        '/home/sippycups/absa/sips/data/nfl/nfl_game_info.csv')
    df = df.fillna(np.nan)

    pattern = r'((?P<temp>-?\d+)\sdegrees)|((?P<humidity>\d+)\%)|(?P<no_wind>\bno wind)|(\bwind (?P<wind_mph>\d+)\smph)|(\bwind chill (?P<wind_chill>-?\d+))'
    
    ret = df.Weather.str.extractall(pattern)
    ret.drop([0, 2, 5, 7], axis=1, inplace=True)
    ret = ret.groupby(level=0).first()
    df = df.join(ret)
    ret.drop('Weather', axis=1, inplace=True)
    return df

def weather():
    df, ret = get_extract()

    return ret

def fix_levels(df, level=0):
    ret = []
    cols = list(df.columns)
    new = pd.DataFrame([], columns=cols)

    indx = df.index
    for elt in indx:
            ret.append(elt[level])
    # for i in ret:

    #     new.append()
    return ret

# for each outermost index of extractall weather df
# 1. update 



def all_infos():
    dfs = {}

    fns = glob.glob(
        '/home/sippycups/absa/sips/data/nfl/games/**team_stats.csv')
    for i, fn in enumerate(fns):
        game_id = utils.path_to_id(fn)
        try:
            df = pd.read_csv(fn)
            # fixed = fix_game_info(df)
            fixed = transpose_fix(df)
        except:
            # print(f'{i}: {game_id} skipped')
            continue

        fixed['game_id'] = game_id
        print(f'{i} {game_id}')

        dfs[game_id] = fixed
    return dfs


# def all_infos2():
#     # to test performance
#     # dfs = {}
#     dfs = []

#     fns = glob.glob(
#         '/home/sippycups/absa/sips/data/nfl/games/**team_stats.csv')
#     for i, fn in enumerate(fns):
#         game_id = utils.path_to_id(fn)
#         try:
#             df = pd.read_csv(fn)
#         except:
#             continue

#         df['game_id'] = game_id
#         dfs.append(df)

#     return list(map(transpose_fix, dfs))

if __name__ == "__main__":
    t0 = time.time()
    dfs = all_infos()
    t1 = time.time()

    # dfs2 = all_infos2()
    # t2 = time.time()

    print(f'delta0 : {t1 - t0}')
    # print(f'delta1 : {t2 - t1}')
    biggo = pd.concat(dfs.values())

    print(biggo)
