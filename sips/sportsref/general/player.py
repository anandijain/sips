import os
import pandas as pd

import sips
from sips.macros import sports_ref as sref
from sips.sportsref import utils as sru
from sips.h import grab
from sips.h import parse


def player(player_url: str, table_ids: list, output="dict", verbose=False):
    dfs = {}

    p = grab.comments(player_url, verbose=False)
    dfs_count = 0
    for t_id in table_ids:
        df = parse.get_table(p, t_id, to_pd=True)
        if df is None:
            continue
        dfs[t_id] = df
        dfs_count += 1

    if verbose:
        p_id = sru.url_to_id(player_url)
        print(f"{p_id} : {dfs_count}")

    if output == "list":
        dfs = list(dfs.values())

    return dfs


def players(sport: str, table_ids: list):

    path = sips.PARENT_DIR + "data/" + sport + "/players/"
    links_df = pd.read_csv(path + "index.csv")

    if sport == "nba":
        links = sref.nba_no_slash + links_df.link
    elif sport == "nfl":
        links = sref.nfl_no_slash + links_df.link
    else:
        links = links_df.link

    if not os.path.isdir(path):
        os.mkdir(path)

    for i, link in enumerate(links):
        p_id = sru.url_to_id(link)
        player_path = path + p_id + "/"

        if not os.path.isdir(player_path):
            os.mkdir(player_path)

        dfd = player(link, table_ids)

        df_count = 0
        for t_id, df in dfd.items():
            if df is None:
                continue
            fn = p_id + "_" + t_id
            df.to_csv(player_path + fn + ".csv")
            df_count += 1

        print(f"{i}: {link}: {df_count}")


if __name__ == "__main__":
    pass
