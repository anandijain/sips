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

    mapping = {
        "line_score": line_score(fn),
        "four_factors": four_factors(fn),
        "basic": game_basic(fn),
        "advanced": game_advanced(fn),
        "pbp": game_pbp(fn),
    }
    for m in mapping.keys():
        if m in fn:
            df = mapping[m](fn)
    return df


def line_score(fn):
    return utils.drop_rename_from_fn(
        fn, nba.LINE_SCORES, drop_n=1)


def all_four_factors(path='', write=True):
    basic_fns = glob.glob(path + '**four_factors.csv')
    # skipped = []
    # dfs = []
    for i, fn in enumerate(basic_fns):
        try:
            df = four_factors(fn)
            if df is not None:
                # dfs.append(df)
                print(f'{i}: {fn}')
                if write:
                    df.to_csv(fn)
        except:
            # skipped.append(utils.path_to_id(fn))
            continue
    # return dfs, skipped


def four_factors(fn):
    # print(fn)
    try:
        df = utils.drop_rename_from_fn(
            fn, nba.FOUR_FACTORS, id_col='game_id', drop_n=1)
        return df
    except:
        return


def game_pbp(fn):
    df = utils.drop_rename_from_fn(fn, nba.GAME_PBP, drop_n=1)
    df = game_pbp_times(df)
    return df


def all_game_basic(path='', write=True):
    basic_fns = glob.glob(path + '**game-basic.csv')
    # dfs = []
    # skipped = []

    for i, fn in enumerate(basic_fns):
        try:
            df = game_basic(fn)
            if df is not None:
                # dfs.append(df)
                print(f'{i}: {fn}')
                if write:
                    df.to_csv(fn)
        except:
            # skipped.append(utils.path_to_id(fn))
            continue
    # return dfs, skipped


def game_basic(fn):
    try:
        df = utils.drop_rename_from_fn(fn, nba.GAME_BASIC, drop_n=1)
        return df
    except:
        return


def game_advanced(fn):
    df = utils.drop_rename_from_fn(fn, nba.GAME_ADVANCED, drop_n=1)
    return df


def clean_player_file(fn: str, verbose=False):
    table_name = utils.player_table_type(fn)
    try:
        drop_n = nba.PLAYER_DROP_N[table_name]
        cols = nba.PLAYER_TABLES[table_name]
        df = utils.drop_rename_from_fn(
            fn, cols, id_col='player_id', drop_n=drop_n)
    except:
        df = pd.read_csv(fn)
    return df


def shotchart_tip(df: pd.DataFrame):
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


def game_pbp_times(df: pd.DataFrame):
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


def add_total_time_remaining(df: pd.DataFrame, new_col='tot_sec', qtr='qtr', mins='mins', secs='secs'):
    # 720 seconds in nba qtr, 4 qtrs
    df[new_col] = 720*(4 - df[qtr]) + df[mins] * 60 + df[secs]
    return df


def lines_tot_time(df: pd.DataFrame):
    df = df[df.qtr != 'None']
    df = df[df.secs != 'None']
    df[['qtr', 'secs']] = df[['qtr', 'secs']].astype(int)
    df.secs = df.secs.replace(-1, 0)
    df['tot_sec'] = 720*(4 - df['qtr']) + df['secs']
    return df


def salaries(season: str = None, write=False, fn='player_salaries.csv', verbose=False):
    fns = glob.glob(m.NBA_PLAYER_DATA + '**salaries.csv')
    d = {utils.path_to_id(fn): pd.read_csv(fn) for fn in fns}
    for p_id, df in d.items():
        df['player_id'] = p_id
    df = pd.concat(d.values())
    df.Salary = df.Salary.apply(utils.sal_to_int)

    if season:
        df = df[df.Season == season]

    if write:
        df.to_csv(fn)

    if verbose:
        print(df)

    return df


def gen_cols():
    a_reserves, h_reserves = [[team_pre + '_reserve' +
                               str(i) for i in range(1, 11)] for team_pre in ['A', 'H']]

    a_starters, h_starters = [[team_pre + '_starter' +
                               str(i) for i in range(1, 6)] for team_pre in ['A', 'H']]
    return a_reserves, h_reserves, a_starters, h_starters

def players_and_teams():
    fns = ['nba_advanced_player_history.csv', 'nba_history_wt_and_p.csv']
    path = '/home/sippycups/absa/sips/data/misc/'
    num_cols_team = 15 * 17
    player_data, ids_stats_og = [pd.read_csv(path + fn) for fn in fns]
    a_reserves, h_reserves, a_starters, h_starters = gen_cols()

    union_ids = player_data.Game_id.unique()
    ids_stats_df = ids_stats_og[ids_stats_og.Game_id.isin(union_ids)]

    a_pids = a_starters + a_reserves
    h_pids = h_starters + h_reserves

    p_id_cols = a_pids + h_pids
    p_id_cols.append('Game_id')
    ids_df = ids_stats_df[p_id_cols]
    drop_cols = ['Date', 'MP', 'Players', 'Game_id']
    all_ids = ids_df.drop('Game_id', axis=1)
    gs_dict = {k: v for k, v in player_data.groupby('Game_id')}
    stuff = []
    for (idx, row) in ids_stats_df.iterrows():
        g_id = row.Game_id
        g = gs_dict[g_id]
        
        dropped_g = g.drop(drop_cols, axis=1)

        row_ids = row[p_id_cols].drop('Game_id')

        a_ids = row_ids[a_pids]
        h_ids = row_ids[h_pids]

        a_stats = g[g.Players.isin(a_ids)]
        h_stats = g[g.Players.isin(h_ids)]
        
        h_data = h_stats.drop(drop_cols, axis=1).values
        a_data = a_stats.drop(drop_cols, axis=1).values

        h_data = pad(h_data, num_cols_team)
        a_data = pad(a_data, num_cols_team)
        if h_data is None or a_data is None:
            continue
        to_add = list(np.concatenate([a_data, h_data]))
        to_add.append(g_id)
        stuff.append(to_add)

    to_merge = pd.DataFrame(stuff)
    
    df = pd.merge(ids_stats_og, to_merge, how='inner', left_on='Game_id', right_on=510)

    return df



def pad(x, nzeros=255):
    try:
        result = np.zeros(nzeros)
        result[0:x.size] = x.reshape(x.size)
        return result
    except ValueError:
        return 
    


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

    # df = test_pbp()
    # print(df.a_pts.unique())
    # print(df.dtypes)

    # df = salaries(season='Career', write=True, fn='player_career_sals.csv')
    # all_four_factors(path='/home/sippycups/absa/sips/data/nba/games/')
    # all_game_basic(path='/home/sippycups/absa/sips/data/nba/games/')
#
    # print(df)

    df = players_and_teams()
    print(df)
