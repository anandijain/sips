import glob
from functools import reduce

import pandas as pd
from sips.sportsref.nba_ref import cleaners
from sips.sportsref import utils
from sips.macros import macros as m

GAME_DATA = m.PARENT_DIR + "data/nba/games/"


LINES = '/home/sippycups/absa/sips/data/lines/lines/6584352.csv'
GAME_ID = '202001070LAL'


def shots_fns(game_id, folder=GAME_DATA):
    home_team = utils.game_id_to_home_code(game_id)
    charts_fns = glob.glob(f'{folder}{game_id}**shotchart.csv')
    away_shots_fn = glob.glob(
        f'{folder}{game_id}_{home_team}**shotchart.csv')[0]
    home_shots_fn = charts_fns.remove(away_shots_fn)

    return home_shots_fn, away_shots_fn


def to_sync_dfs(game_id):
    lines = pd.read_csv(LINES)

    home_shots, away_shots = shots_fns(game_id)

    pbp_fn = game_id + '_pbp.csv'

    pbp, charth, charta = [pd.read_csv(GAME_DATA + fn)
                           for fn in [pbp_fn, home_shots, away_shots]]
    return lines, pbp, charth, charta


def sync_shots(home, away):
    home = cleaners.shotchart_tip(home)
    home['home'] = 1
    away = cleaners.shotchart_tip(away)
    away['home'] = 0
    shots = pd.concat([home, away])
    shots.drop(shots.columns[0], axis=1, inplace=True)
    shots.sort_values(by='tot_sec', ascending=False, inplace=True)
    return shots


def sync(how='inner'):
    game_id = '201912280MIL'
    lines, pbp, sch, sca = to_sync_dfs(game_id)
    shots = sync_shots(sch, sca)

    pbp = cleaners.drop_rename(pbp, pbp.iloc[0], drop_n=1)
    pbp = cleaners.game_pbp_times(pbp)

    lines.rename(columns={'quarter': 'qtr'}, inplace=True)
    lines = cleaners.lines_tot_time(lines)

    synced = reduce(lambda l, r: pd.merge(
        l, r, on='tot_sec', how=how), [shots, lines, pbp])
    synced['game_id'] = game_id
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
