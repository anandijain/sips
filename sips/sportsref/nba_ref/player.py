import os
import pandas as pd

import sips
from sips.macros import sports_ref as sref
from sips.h import grab
from sips.h import parse
from sips.sportsref import player
from sips.sportsref import utils as sru


table_ids = [
    "per_game",
    "totals",
    "per_minute",
    "per_poss",
    "advanced",
    "shooting",
    "pbp",
    "year-and-career-highs",
    "playoffs_per_game",
    "playoffs_totals",
    "playoffs_per_minute",
    "playoffs_per_poss",
    "playoffs_advanced",
    "playoffs_shooting",
    "playoffs_pbp",
    "year-and-career-highs-po",
    "all_star",
    "sim_thru",
    "all_college_stats",
    "sim_career",
    "all_salaries",
    "contracts_orl",
]


def player_links(write=True):
    all_players = []
    # all_players = pd.DataFrame()

    links = [sref.nba_url + "players/" + letter for letter in sref.letters]
    ps = grab.pages(links, output="dict")

    for i, (l, p) in enumerate(ps.items()):
        t = parse.get_table(p, "players")

        rows = t.find_all("tr")
        rows.pop(0)  # header

        p_ids = []
        section_count = 0
        for row in rows:
            th = row.th
            if not th:
                continue
            p_id = th.get("data-append-csv")
            if not p_id:
                continue
            a_tag = th.find("a")
            link = a_tag.get("href")
            if not link:
                continue
            section_count += 1
            all_players.append(link)

        print(f"{i} : {sref.letters[i]} : {section_count}")

    if write:
        df = pd.DataFrame(all_players, columns=["link"])
        df.to_csv(sips.PARENT_DIR + "data/players/index.csv")

    return all_players


if __name__ == "__main__":
    player.players("nba", table_ids)
