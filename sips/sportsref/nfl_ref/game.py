import pandas as pd

from sips.h import grab
from sips.h import parse
from sips.macros import sports_ref as sref
from sips.macros import macros as m
from sips.sportsref import utils as sru
from sips.sportsref import player
from sips.sportsref import boxlinks as bl

table_ids = [
    "scoring",
    "game_info",
    "officials",
    "expected_points",
    "team_stats",
    "player_offense",
    "player_defense",
    "returns",
    "kicking",
    "passing_advanced",
    "rushing_advanced",
    "receiving_advanced",
    "defense_advanced",
    "home_starters",
    "vis_starters",
    "home_snap_counts",
    "vis_snap_counts",
    "home_drives",
    "vis_drives",
    "pbp",
]


pregame_table_ids = [
    "team_leaders",
    "last_matchups",
    "teams_ranks",
    "chi_current_injuries",
    "dal_current_injuries",
]


def gen_week_links(start=2018, end=1940):
    return [sref.URLS['nfl'] + f"years/{year}/games.htm" for year in range(start, end, -1)]

def nfl_boxlinks(write=True):
    all_links = []
    ls = gen_week_links()
    for i, l in enumerate(ls):
        boxes = bl.boxlinks_from_table(l, data_stat='boxscore_word')
        all_links += boxes
        print(f'{i}: {l} had {len(boxes)} games')

    df = pd.DataFrame(all_links, columns=['game_id'])
    if write:
        folder = sru.gamedata_path('nfl')
        fn = 'index.csv'
        df.to_csv(folder + fn)
    return df

if __name__ == "__main__":

    x = nfl_boxlinks()
    print(x)
    # dfs = player.player(
    #     "https://www.pro-football-reference.com/boxscores/201809090gnb.htm",
    #     table_ids=table_ids,
    # )
    # for k, df in dfs.items():
    #     print(df)
