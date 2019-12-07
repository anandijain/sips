from sips.h import grab
from sips.h import parse
from sips.sportsref import utils as sru
from sips.sportsref import player

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

if __name__ == "__main__":
    dfs = player.player(
        "https://www.pro-football-reference.com/boxscores/201809090gnb.htm",
        table_ids=table_ids,
    )
    for k, df in dfs.items():
        print(df)
