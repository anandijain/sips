import os
import pandas as pd

import sips
from sips.macros import sports_ref as sref
from sips.h import grab
from sips.h import parse
from sips.sportsref.general import player
from sips.sportsref import utils as sru


def player_links():
    all_players = []

    links = [sref.nhl_url + "players/" + letter for letter in sref.letters]

    ps = grab.get_pages(links)

    for i, p in enumerate(ps):
        print(f"{i}")
        div = p.find("div", {"id": "div_players"})
        if not div:
            continue
        a_tags = div.find_all("a")
        p_links = [sref.nhl_no_slash + a_tag["href"] for a_tag in a_tags if a_tag]
        all_players += p_links

    df = pd.DataFrame(all_players, columns=["link"])
    df.to_csv(sips.PARENT_DIR + "data/nhl/players/index.csv")
    return df


if __name__ == "__main__":
    player_links()
