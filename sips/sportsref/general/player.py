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


if __name__ == "__main__":
    pass
