import os
import pandas as pd

import sips
from sips.macros import sports_ref as sref
from sips.h import grab
from sips.h import parse
from sips.sportsref.general import player
from sips.sportsref import utils as sru

comment_idxs = {
    25: "per_minute",
    26: "per_poss",
    30: "advanced",
    31: "shooting",
    32: "pbp",
    33: "year-and-career-highs",
    34: "playoffs_per_game",
    35: "playoffs_totals",
    36: "playoffs_per_minute",
    37: "playoffs_per_poss",
    38: "playoffs_advanced",
    39: "playoffs_shooting",
    40: "playoffs_pbp",
    41: "year-and-career-highs-po",
    42: "all_star",
    43: "sim_thru",
    44: "sim_career",
}


def player_links(write=False):
    all_players = []
    # all_players = pd.DataFrame()

    links = [sref.bk_url + "players/" + letter for letter in sref.letters]
    ps = grab.get_pages(links)
    for p in ps:
        t = parse.get_table(p, "players")
        print(len(t))
        # df = [pd.read_html(t.prettify())[0] for e in t]
        ths = t.find_all("th", {"data-stat": "player"})
        print(ths)
        p_ids = []
        for th in ths:
            a_tags = th.find_all("a")
            if a_tags is not None:
                for a_tag in a_tags:
                    pl_id = a_tag["href"]
                    print(pl_id)
                    all_players.append(pl_id)
        # df['ids'] = pd.Series(p_ids)
        # all_players = pd.concat([all_players, df])
    if write:
        df = pd.DataFrame(all_players, columns=["link"])
        df.to_csv(sips.PARENT_DIR + "data/players/index.csv")
    return all_players


if __name__ == "__main__":
    # sfx = '/players/j/jamesle01.html' sfx
    table_ids = ["per_game", "totals"]
    links = player_links()
    path = sips.PARENT_DIR + "data/players/"
    if not os.path.isdir(path):
        os.mkdir(path)
    # print(links)
    for i, link in enumerate(links):
        # if link == "Player" or link == "From" or link == "To":
        #     continue
        player_url = sref.bk_no_slash + link
        p_id = sru.url_to_id(player_url)
        player_path = path + p_id + "/"
        print(f"{i}: {player_url}")

        if not os.path.isdir(player_path):
            os.mkdir(player_path)

        dfd = player.player(player_url, table_ids, comment_idxs)

        for t_id, df in dfd.items():
            if not df:
                continue
            df = df[0]
            fn = p_id + "_" + t_id
            df.to_csv(player_path + fn + ".csv")

    # lebron = player(sfx)
    # print(lebron)
