import os
import glob
import types


import pandas as pd
import numpy as np
from sips.macros.sports import nba
from sips.macros import macros as m
from sips.sportsref import utils

TEST_GAME_ID = '202001220ORL'
GAME_TEST_FILES = glob.glob(m.NBA_GAME_DATA + TEST_GAME_ID + '*')


TEST_PLAYER_ID = 'curryst01'
PLAYER_TEST_FILES = glob.glob(m.NBA_PLAYER_DATA + TEST_PLAYER_ID + '*')


def clean_games(write=False):
    fns = os.listdir(m.NBA_GAME_DATA)
    clean_games_files(fns, write=write)


def clean_games_files(fns: list, write=False):
    try:
        fns.remove("index.csv")
    except ValueError:
        pass
    for i, fn in enumerate(fns):
        full_fn = m.NBA_GAME_DATA + fn

        df = clean_game_file(full_fn)

        if df is not None:
            print(f"{i}: {fn} cleaned")
        else:
            print(f"{i}: {fn} skipped")
        if write:
            df.to_csv(full_fn)


def clean_game_file(fn: str, verbose=False):
    if "line_score" in fn:
        df = utils.drop_rename_from_fn(
            fn, nba.LINE_SCORES, drop_n=1, verbose=verbose)
    elif "four_factors" in fn:
        df = utils.drop_rename_from_fn(fn, nba.FOUR_FACTORS,
                                 drop_n=1, verbose=verbose)
    elif "basic" in fn:
        df = utils.drop_rename_from_fn(fn, nba.GAME_BASIC, drop_n=1, verbose=verbose)
    elif "advanced" in fn:
        df = utils.drop_rename_from_fn(
            fn, nba.GAME_ADVANCED, drop_n=1, verbose=verbose)
    elif "pbp" in fn:
        df = utils.drop_rename_from_fn(fn, nba.GAME_PBP, drop_n=1, verbose=verbose)
        df = game_pbp_times(df)
    elif "shooting" in fn:
        # drop first col
        df = drop_ith_col(fn, 0)
    else:
        return
    return df


def clean_player_file(fn:str, verbose=False):
    # im dumb
    if "totals" in fn:
        df = utils.drop_rename_from_fn(
            fn, nba.PLAYER_TOTALS, id_col='player_id', drop_n=0, verbose=verbose)
    # elif "highs-po" in fn:
    #     df = utils.drop_rename_from_fn(
    #         fn, nba.PLAYER_YEAR_CAREER_HIGHS_PO, id_col='player_id', drop_n=0, verbose=verbose)
    elif "highs" in fn:
        df = utils.drop_rename_from_fn(
            fn, nba.PLAYER_YEAR_CAREER_HIGHS, id_col='player_id', drop_n=1, verbose=verbose)
    elif "per_minute" in fn:
        df = utils.drop_rename_from_fn(
            fn, nba.PLAYER_PER_MIN, id_col='player_id', drop_n=0, verbose=verbose)
    elif "per_game" in fn:
        df = utils.drop_rename_from_fn(
            fn, nba.PLAYER_PER_GAME, id_col='player_id', drop_n=0, verbose=verbose)
    elif "advanced" in fn:
        df = utils.drop_rename_from_fn(
            fn, nba.PLAYER_ADVANCED, id_col='player_id', drop_n=0, verbose=verbose)
    elif "salaries" in fn:
        df = utils.drop_rename_from_fn(
            fn, nba.PLAYER_SALARIES, id_col='player_id', drop_n=0, verbose=verbose)
    elif "college" in fn:
        df = utils.drop_rename_from_fn(
            fn, nba.PLAYER_COLLEGE, id_col='player_id', drop_n=1, verbose=verbose)
    elif "shooting" in fn:
        df = utils.drop_rename_from_fn(
            fn, nba.PLAYER_SHOOTING, id_col='player_id', drop_n=2, verbose=verbose)
    elif "pbp" in fn:
        df = utils.drop_rename_from_fn(
            fn, nba.PLAYER_PBP, id_col='player_id', drop_n=1, verbose=verbose)
    elif "sim_thru" in fn:
        df = utils.drop_rename_from_fn(
            fn, nba.PLAYER_SIM_THRU, id_col='player_id', drop_n=1, verbose=verbose)
    elif "sim_career" in fn:
        df = utils.drop_rename_from_fn(
            fn, nba.PLAYER_SIM_CAREER, id_col='player_id', drop_n=1, verbose=verbose)
    else:
        return
    return df




def shotchart_tip(df:pd.DataFrame):
    # new_cols = ["index", "p_id", "qtr", "shot_made", "x_pos", "y_pos", "time", 'player_info', 'team_info']
    tips = df.tip
    tmp = tips.str.split('<br>', expand=True)

    time_remaining = tmp[0].str.split(' ', expand=True)[2]
    ms = split_str_times(time_remaining)
    df['mins'] = ms[0]
    df['secs'] = ms[1]

    df = add_total_time_remaining(df)

    df['player_info'] = tmp[1]
    df['team_info'] = tmp[2]
    df = df.drop('tip', axis=1)
    return df


def split_str_times(times: pd.Series):
    ms = times.str.split(':', expand=True).astype(np.float)
    return ms





def game_pbp_times(df:pd.DataFrame):
    # already correct colnames
    # t = df.Time
    q2_idx = df.index[df.Time == '2nd Q'][0]
    q3_idx = df.index[df.Time == '3rd Q'][0]
    q4_idx = df.index[df.Time == '4th Q'][0]
    # q4_idx = df.index[df.Time == '1st OT'][0] # TODO

    s = pd.Series(np.full(q2_idx, 1))
    s = pd.concat([s, pd.Series(np.full(q3_idx - q2_idx, 2))])
    s = pd.concat([s, pd.Series(np.full(q4_idx - q3_idx, 3))])
    # s = pd.concat([s, pd.Series(np.full(len(df.Time) - q4_idx, 4))])
    s = pd.concat([s, pd.Series(np.full(len(df.Time) - q4_idx, 4))])
    s = s.reset_index(drop=True)
    df['qtr'] = s
    bad_strs = ['1st Q', '2nd Q', '3rd Q', '4th Q', 'Time', '1st OT', '2nd OT']
    for s in bad_strs:
        df = df[df.Time != s]

    scores = df.score.str.split('-', expand=True)
    print(scores)
    df[['a_pts', 'h_pts']] = scores
    df.drop('score', axis=1, inplace=True)
    df = utils.split_str_times_df(df, col='Time')
    df = add_total_time_remaining(df)
    return df




def add_total_time_remaining(df:pd.DataFrame, new_col='tot_sec', qtr='qtr', mins='mins', secs='secs'):
    # 720 seconds in nba qtr, 4 qtrs
    df[new_col] = 720*(4 - df[qtr]) + df[mins] * 60 + df[secs]
    return df


def lines_tot_time(df:pd.DataFrame):
    df = df[df.qtr != 'None']
    df = df[df.secs != 'None']
    df[['qtr', 'secs']] = df[['qtr', 'secs']].astype(int)
    df.secs = df.secs.replace(-1, 0)
    df['tot_sec'] = 720*(4 - df['qtr']) + df['secs']
    return df


def test_player():
    for f in PLAYER_TEST_FILES:
        df = clean_player_file(f, verbose=True)
        if df is not None:
            print(f'post: {df}')
        else:
            print(f)


def test_game():
    for f in GAME_TEST_FILES:
        df = clean_game_file(f, verbose=True)
        print(f)
        if df is not None:
            print(f'post: {df}')


def test_pbp():
    fn = '201611120DEN_pbp.csv'
    df = clean_game_file(m.NBA_GAME_DATA + fn)
    print(df)
    return df


if __name__ == "__main__":
    # test_player()
    # test_game()
    df = test_pbp()
    print(df.a_pts.unique())
    print(df.dtypes)
