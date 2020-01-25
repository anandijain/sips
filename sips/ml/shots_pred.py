import glob
import pandas as pd

from sips.sportsref.nba_ref import cleaners as cl
from sips.macros import macros as m

GAME_DATA = m.PARENT_DIR + "data/nba/games/"


files = glob.glob(GAME_DATA + '*shotchart.csv')


def shotchart(fn):
    df = pd.read_csv(fn)
    df = df.drop(df.columns[0], axis=1)
    g_id = cl.full_fn_to_game_id(fn)
    df['game_id'] = g_id
    return df


def compile_shots():
    dfs = []
    for i, f in enumerate(files):
        df = shotchart(f)
        dfs.append(df)
        if i % 50 == 0:
            print(f'{i} {f}')
    return dfs


if __name__ == "__main__":
    dfs = compile_shots()
