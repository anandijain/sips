import pandas as pd

import sips
from sips.macros import sports_ref as sref
from sips.h import grab
from sips.h import parse
from sips.sportsref import player
from sips.sportsref import utils as sru
from sips.macros import macros as m


def player_links(write=True):
    """

    """
    player_rows = []
    # all_players = pd.DataFrame()

    links = [sref.NBA_URL + "players/" + letter for letter in sref.LETTERS]
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
            name = a_tag.text
            section_count += 1
            player_rows.append([name, p_id, link])
        print(f"{i} : {sref.LETTERS[i]} : {section_count}")

    all_players = pd.DataFrame(player_rows, columns=["name", "id", "link"])

    if write:
        all_players.to_csv(m.PARENT_DIR + "data/nba/players/index.csv")

    return all_players


if __name__ == "__main__":
    df = player_links(write=True)
    print(df)
    # player.players("nba", sref.TABLE_IDS['nba'])
