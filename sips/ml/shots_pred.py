import glob
import pandas as pd

from sips.sportsref.nba_ref import cleaners as cl
from sips.macros import macros as m

GAME_DATA = m.PARENT_DIR + "data/nba/games/"



LINES = '/home/sippycups/absa/sips/data/lines/lines/6584352.csv'
GAME_ID = '202001070LAL'

def sync():
    lines = pd.read_csv(LINES)
    pbp_fn = '201912280MIL_pbp.csv'
    chart_fn = '201912280MIL_MIL_shotchart.csv'
    chart_fn2 = '201912280MIL_ORL_shotchart.csv'
    pbp, charth, charta = [pd.read_csv(GAME_DATA + fn) for fn in [pbp_fn, chart_fn, chart_fn2]]
    return lines, pbp, charth, charta

def shotchart(fn):
    df = pd.read_csv(fn)
    df = df.drop(df.columns[0], axis=1)
    g_id = cl.full_fn_to_game_id(fn)
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
    ldf, gfs = sync()

    # dfs = compile_shots()
