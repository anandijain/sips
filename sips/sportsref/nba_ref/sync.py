import os
import glob
from functools import reduce

import pandas as pd
from sips.sportsref.nba_ref import cleaners
from sips.sportsref import utils
from sips.macros import macros as m
from sips.macros.sports import nba

GAME_DATA = m.PARENT_DIR + "data/nba/games/"


LINES = '/home/sippycups/absa/sips/data/lines/lines/6584352.csv'
GAME_ID = '202001070LAL'


def check_chart_files(game_id: str, folder=GAME_DATA) -> bool:
    glob_charts = glob.glob(f'{folder + game_id}**shotchart.csv')
    if len(glob_charts) < 2:
        return False
    else:
        return True


def check_pbp_file(game_id: str, folder=GAME_DATA) -> bool:
    pbp_fn = folder + game_id + '_pbp.csv'
    return os.path.isfile(pbp_fn)


def shots_fns(game_id, folder=GAME_DATA):
    """

    by now we should have already verified all 
    """
    home_team = utils.game_id_to_home_code(game_id)

    glob_charts = f'{folder + game_id}**shotchart.csv'
    charts_fns = glob.glob(glob_charts)

    home_shots_fn = glob.glob(
        f'{folder + game_id}_{home_team}**shotchart.csv')[0]

    charts_fns.remove(home_shots_fn)
    away_shots_fn = charts_fns[0]

    return home_shots_fn, away_shots_fn


def to_sync_dfs(game_id):

    shots = shots_fns(game_id)
    if shots is None:
        return
    home_shots_fn, away_shots_fn = shots
    pbp_fn = GAME_DATA + game_id + '_pbp.csv'

    try:
        pbp, charth, charta = [pd.read_csv(fn)
                               for fn in [pbp_fn, home_shots_fn, away_shots_fn]]
    except FileNotFoundError:
        print(f'{game_id} missing data')
        return

    return pbp, charth, charta


def sync_shots(home, away):
    home = cleaners.shotchart_tip(home)
    home['home'] = 1
    away = cleaners.shotchart_tip(away)
    away['home'] = 0
    shots = pd.concat([home, away])
    shots.drop(shots.columns[0], axis=1, inplace=True)
    shots.sort_values(by='tot_sec', ascending=False, inplace=True)
    return shots


def sync(game_id, how='inner'):
    dfs = to_sync_dfs(game_id)
    if dfs is not None:
        pbp, sch, sca = dfs
    else:
        return
    shots = sync_shots(sch, sca)

    pbp = cleaners.drop_rename(pbp, nba.GAME_PBP, drop_n=1)
    pbp = cleaners.game_pbp_times(pbp)

    synced = reduce(lambda l, r: pd.merge(
        l, r, on='tot_sec', how=how), [shots, pbp])
    synced['game_id'] = game_id
    return synced


def sync_all(n: int = 50, how='inner') -> dict:
    synced = {}
    synced_count = 0

    samples = get_shotchart_ids()

    for i, game_id in enumerate(samples.sc_game_ids):
        if not check_chart_files(game_id) or not check_pbp_file(game_id):
            continue
        if synced_count == n:
            break

        game = sync(game_id, how=how)
        synced[game_id] = game
        print(f'{i}: {game_id} synced. # synced: {synced_count} ')
        synced_count += 1

    return synced


def shotchart(fn: str) -> pd.DataFrame:
    df = pd.read_csv(fn)
    df = df.drop(df.columns[0], axis=1)
    g_id = cleaners.full_fn_to_game_id(fn)
    df['game_id'] = g_id
    return df


def compile_all_shotcharts() -> list:
    files = glob.glob(GAME_DATA + '*shotchart.csv')
    dfs = []
    for i, f in enumerate(files):
        df = shotchart(f)
        dfs.append(df)
        if i % 50 == 0:
            print(f'{i} {f}')
    return dfs


def get_shotchart_ids():
    files = glob.glob(GAME_DATA + '*shotchart.csv')
    ids = pd.Series(
        list(map(glob_to_id, files)), )
    df = pd.DataFrame(ids.unique(), columns=['sc_game_ids'])
    df.sort_values(by='sc_game_ids', ascending=False, inplace=True)
    return df


def glob_to_id(glob_path: str):
    return glob_path.split('_')[0].split('/')[-1]


if __name__ == "__main__":
    s = sync_all()
    print(s)
