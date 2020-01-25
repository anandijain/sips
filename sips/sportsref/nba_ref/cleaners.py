import os
import glob
import types


import pandas as pd

from sips.macros.sports import nba

GAME_DATA = "/home/sippycups/absa/sips/data/nba/games/"
TEST_GAME_ID = '202001220ORL'
GAME_TEST_FILES = glob.glob(GAME_DATA + TEST_GAME_ID + '*')


PLAYER_DATA = "/home/sippycups/absa/sips/data/nba/players/"
TEST_PLAYER_ID = 'curryst01'
PLAYER_TEST_FILES = glob.glob(PLAYER_DATA + TEST_PLAYER_ID + '*')


def clean_games(write=False):
    fns = os.listdir(GAME_DATA)
    clean_games_files(fns, write=write)


def clean_games_files(fns: list, write=False):
    try:
        fns.remove("index.csv")
    except ValueError:
        pass
    for i, fn in enumerate(fns):
        full_fn = GAME_DATA + fn

        df = clean_game_file(full_fn)

        if df is not None:
            print(f"{i}: {fn} cleaned")
        else:
            print(f"{i}: {fn} skipped")
        if write:
            df.to_csv(full_fn)


def clean_game_file(fn:str, verbose=False):
    if "line_score" in fn:
        df = drop_rename_from_fn(
            fn, nba.LINE_SCORES, drop_n=1, verbose=verbose)
    elif "four_factors" in fn:
        df = drop_rename_from_fn(fn, nba.FOUR_FACTORS,
                                 drop_n=1, verbose=verbose)
    elif "basic" in fn:
        df = drop_rename_from_fn(fn, nba.GAME_BASIC, drop_n=1, verbose=verbose)
    elif "advanced" in fn:
        df = drop_rename_from_fn(
            fn, nba.GAME_ADVANCED, drop_n=1, verbose=verbose)

    elif "shooting" in fn:
        # drop first col
        df = drop_ith_col(fn, 0)
    else:
        return
    return df


def clean_player_file(fn, verbose=False):
    # im dumb
    if "totals" in fn:
        df = drop_rename_from_fn(
            fn, nba.PLAYER_TOTALS, id_col='player_id', drop_n=0, verbose=verbose)
    # elif "highs-po" in fn:
    #     df = drop_rename_from_fn(
    #         fn, nba.PLAYER_YEAR_CAREER_HIGHS_PO, id_col='player_id', drop_n=0, verbose=verbose)
    elif "highs" in fn:
        df = drop_rename_from_fn(
            fn, nba.PLAYER_YEAR_CAREER_HIGHS, id_col='player_id', drop_n=1, verbose=verbose)
    elif "per_minute" in fn:
        df = drop_rename_from_fn(
            fn, nba.PLAYER_PER_MIN, id_col='player_id', drop_n=0, verbose=verbose)
    elif "per_game" in fn:
        df = drop_rename_from_fn(
            fn, nba.PLAYER_PER_GAME, id_col='player_id', drop_n=0, verbose=verbose)
    elif "advanced" in fn:
        df = drop_rename_from_fn(
            fn, nba.PLAYER_ADVANCED, id_col='player_id', drop_n=0, verbose=verbose)
    elif "salaries" in fn:
        df = drop_rename_from_fn(
            fn, nba.PLAYER_SALARIES, id_col='player_id', drop_n=0, verbose=verbose)
    elif "college" in fn:
        df = drop_rename_from_fn(
            fn, nba.PLAYER_COLLEGE, id_col='player_id', drop_n=1, verbose=verbose)
    elif "shooting" in fn:
        df = drop_rename_from_fn(
            fn, nba.PLAYER_SHOOTING, id_col='player_id', drop_n=2, verbose=verbose)
    elif "pbp" in fn:
        df = drop_rename_from_fn(
            fn, nba.PLAYER_PBP, id_col='player_id', drop_n=1, verbose=verbose)
    elif "sim_thru" in fn:
        df = drop_rename_from_fn(
            fn, nba.PLAYER_SIM_THRU, id_col='player_id', drop_n=1, verbose=verbose)
    elif "sim_career" in fn:
        df = drop_rename_from_fn(
            fn, nba.PLAYER_SIM_CAREER, id_col='player_id', drop_n=1, verbose=verbose)
    else:
        return
    return df


def drop_rename(df, columns, drop_n=0):
    df = df.drop(range(drop_n))
    df.columns = columns
    if "index" in df.columns:
        df = df.drop("index", axis=1)
    return df


def drop_rename_from_fn(fn, cols, id_col='game_id', drop_n=0, verbose=False):
    df = pd.read_csv(fn)
    if verbose:
        print(f'pre: {df}')
    df = drop_rename(df, cols, drop_n=drop_n)
    df = add_id_from_fn(df, fn, id_col)
    return df


def add_id_from_fn(df, fn, col='player_id'):
    obj_id = full_fn_to_game_id(fn)  # player or game
    df[col] = obj_id
    return df


def key_to_game_id(s: str):
    return s.split("_")[0]


def full_fn_to_game_id(s: str):
    s = key_to_game_id(s)
    return s.split("/")[-1]


def drop_ith_col(fn, i):
    df = pd.read_csv(fn)
    df = df.drop(df.columns[i], axis=1)
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



if __name__ == "__main__":
    # test_player()
    test_game()
