import time
import os
import glob
import types
import re

import pandas as pd
import numpy as np

from sips.sportsref import utils
from sips.macros.sports import nfl
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


def duration_fix(t: pd.Series) -> pd.Series:
    ts = t.str.split(':', expand=True).astype(np.float)

    mins = ts[0] * 60 + ts[1]
    mins.name = 'mins'
    return mins


def weather():
    df = pd.read_csv(
        m.NFL_DATA + 'nfl_game_info.csv')
    df = df.fillna(np.nan)

    pattern = r'((?P<temp>-?\d+)\sdegrees)|((?P<humidity>\d+)\%)|(?P<no_wind>\bno wind)|(\bwind (?P<wind_mph>\d+)\smph)|(\bwind chill (?P<wind_chill>-?\d+))'

    ret = df.Weather.str.extractall(pattern)
    ret.drop([0, 2, 5, 7], axis=1, inplace=True)
    ret = ret.groupby(level=0).first()

    df = df.join(ret)
    df['no_wind'] = df.no_wind.replace('no wind', 1)
    df['no_wind'] = df.no_wind.astype(np.float)
    print(df.columns)
    df.drop('Weather', axis=1, inplace=True)
    return df


def main():
    df = weather()
    mins = duration_fix(df.Duration)
    df['mins'] = mins
    df.drop('Duration', axis=1, inplace=True)
    return df


def scoring():
    table_type = 'scoring'
    dfs = utils.group_read(table_type, sport='nfl')
    new_dfs = []
    for key, df in dfs.items():
        if len(df.columns) == 8:
            df = utils.drop_rename(df, nfl.GAME_SCORING)
            new_dfs.append(df)
    return new_dfs


if __name__ == "__main__":
    dfs = utils.group_read('scoring', sport='nfl')
    dfs = scoring()
    biggo = pd.concat(dfs)

    print(biggo)
