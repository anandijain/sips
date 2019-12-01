import os
import time
import pandas as pd

import sips
from sips.macros import sports_ref as sref
from sips.h import grab
from sips.h import parse
from sips.sportsref.general import player
from sips.sportsref import utils as sru


table_ids = [
    "stats",
    "rushing_and_receiving",
    "detailed_rushing_and_receiving",
    "returns",
    "defense",
    "scoring",
    "snap_counts",
    "all_pro",
    "fantasy",
    "combine",
]


def player_links(output="df", write_df=False):
    all_links = []

    section_links = [
        sref.nfl_url + "players/" + letter.upper()
        for letter in sref.letters.append("X")
    ]

    ps = {l: grab.page(l) for l in section_links}
    for l, p in ps.items():
        div = p.find("div", {"id": "div_players"})
        if not div:
            print(l)
            continue
        a_tags = div.find_all("a")
        links = [a["href"] for a in a_tags]
        all_links += links
    if output == "df":
        all_links = pd.DataFrame(all_links, columns=["link"])
        if write_df:
            all_links.to_csv()
    return all_links


if __name__ == "__main__":
    player.players("nfl", table_ids)
