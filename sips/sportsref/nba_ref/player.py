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


def player_links(write=True):
    all_players = []
    # all_players = pd.DataFrame()

    links = [sref.bk_url + "players/" + letter for letter in sref.letters]
    ps = grab.get_pages(links, output='dict')

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
            a_tag = th.find('a')
            link = a_tag.get('href')
            if not link:
                continue
            section_count += 1
            all_players.append(link)

        print(f'{i} : {sref.letters[i]} : {section_count}')

    if write:
        df = pd.DataFrame(all_players, columns=["link"])
        df.to_csv(sips.PARENT_DIR + "data/players/index.csv")

    return all_players


def main():
    # sfx = '/players/j/jamesle01.html' sfx
    table_ids = ["per_game", "totals"]

    path = sips.PARENT_DIR + "data/nba/players/"
    links_df = pd.read_csv(path + 'index.csv')
    links = links_df.link

    if not os.path.isdir(path):
        os.mkdir(path)

    for i, link in enumerate(links):
        player_url = sref.bk_no_slash + link
        p_id = sru.url_to_id(player_url)
        player_path = path + p_id + "/"

        if not os.path.isdir(player_path):
            os.mkdir(player_path)

        dfd = player.player(player_url, table_ids, comment_idxs)

        df_count = 0
        for t_id, df in dfd.items():
            if not df:
                continue
            df = df[0]
            
            fn = p_id + "_" + t_id
            df.to_csv(player_path + fn + ".csv")
            df_count += 1

        print(f"{i}: {player_url}: {df_count}")


if __name__ == "__main__":
    main()
    # players = player_links()
    # print(df)
    # print(len(players))