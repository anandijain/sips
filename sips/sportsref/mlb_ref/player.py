import os
import pandas as pd

import sips
from sips.macros import sports_ref as sref
from sips.h import grab
from sips.h import parse
from sips.sportsref.general import player
from sips.sportsref import utils as sru


table_ids = [
    "batting_proj",
    "pitching_proj",
    "pitching_standard",
    "pitching_value",
    "pitching_postseason",
    "batting_standard",
    "batting_value",
    "batting_postseason",
    "standard_fielding",
    "appearances",
    "br-salaries",
]


def player_links():
    all_players = []

    links = [sref.mlb_url + "players/" + letter for letter in sref.letters]

    ps = grab.pages(links, output="dict")

    for i, (l, p) in enumerate(ps.items()):
        print(f"{i}: {l}")
        div = p.find("div", {"id": "div_players_"})
        if not div:
            continue
        a_tags = div.find_all("a")
        p_links = [sref.mlb_no_slash + a_tag["href"] for a_tag in a_tags if a_tag]
        all_players += p_links

    df = pd.DataFrame(all_players, columns=["link"])
    df.to_csv(sips.PARENT_DIR + "data/mlb/players/index.csv")
    return df


if __name__ == "__main__":
    player.players("mlb", table_ids)
