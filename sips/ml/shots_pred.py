import glob
from functools import reduce

import pandas as pd
from sips.sportsref.nba_ref import cleaners
from sips.macros import macros as m

GAME_DATA = m.PARENT_DIR + "data/nba/games/"


LINES = '/home/sippycups/absa/sips/data/lines/lines/6584352.csv'
GAME_ID = '202001070LAL'


def to_sync_dfs():
    lines = pd.read_csv(LINES)
    pbp_fn = '201912280MIL_pbp.csv'
    chart_fn = '201912280MIL_MIL_shotchart.csv'
    chart_fn2 = '201912280MIL_ORL_shotchart.csv'
    pbp, charth, charta = [pd.read_csv(GAME_DATA + fn)
                           for fn in [pbp_fn, chart_fn, chart_fn2]]
    return lines, pbp, charth, charta


def sync():
    l, pbp, sch, sca = to_sync_dfs()
    sch = cleaners.shotchart_tip(sch)
    sca = cleaners.shotchart_tip(sca)
    shots = pd.concat([sch, sca])
    shots.drop(shots.columns[0], axis=1, inplace=True)
    shots.sort_values(by='tot_sec', ascending=False, inplace=True)
    pbp = cleaners.drop_rename(pbp, pbp.iloc[0], drop_n=1)
    pbp = cleaners.game_pbp_times(pbp)

    l.rename(columns={'quarter': 'qtr'}, inplace=True)

    l = cleaners.lines_tot_time(l)
    # print(f'shots{shots.columns}')
    # print(f'l{l.columns}')
    # print(f'pbp{pbp.columns}')
    # synced = shots.merge(l, pbp, on='tot_sec')
    synced = reduce(lambda l, r: pd.merge(l, r, on='tot_sec'), [shots, l, pbp])
    return synced


def shotchart(fn):
    df = pd.read_csv(fn)
    df = df.drop(df.columns[0], axis=1)
    g_id = cleaners.full_fn_to_game_id(fn)
    df['game_id'] = g_id
    return df


def compile_shots():
    files = glob.glob(GAME_DATA + '*shotchart.csv')
    dfs = []
    for i, f in enumerate(files):
        df = shotchart(f)
        dfs.append(df)
        if i % 50 == 0:
            print(f'{i} {f}')
    return dfs


if __name__ == "__main__":
    s = sync()
    print(s)
