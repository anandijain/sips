import pandas as pd

from sips.macros import sports_ref as sref

from sips.sportsref import utils 
from sips.sportsref import boxlinks as bl


def gen_nfl_week_links(start=2018, end=1940):
    return [sref.URLS['nfl'] + f"years/{year}/games.htm" for year in range(start, end, -1)]


def nfl_boxlinks(write=True):
    all_links = []
    ls = gen_nfl_week_links()
    for i, l in enumerate(ls):
        boxes = bl.boxlinks_from_url(l, data_stat='boxscore_word')
        all_links += boxes
        print(f'{i}: {l} had {len(boxes)} games')

    df = pd.DataFrame(all_links, columns=['game_id'])
    if write:
        folder = utils.gamedata_path('nfl')
        fn = 'index.csv'
        df.to_csv(folder + fn)
    return df

if __name__ == "__main__":

    x = nfl_boxlinks()
    print(x)
