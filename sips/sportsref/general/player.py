import os
import pandas as pd

import sips
from sips.macros import sports_ref as sref
from sips.h import grab
from sips.h import parse

def player(player_url: str, table_ids: list, comments_dict: dict, output='dict'):
    dfs = {}
    p = grab.get_page(player_url)
    for t_id in table_ids:
        dfs[t_id] = parse.get_table(p, t_id, to_pd=True)

    cs = parse.comments(p)
    for index, t_id in comments_dict.items():
        try:
            c = cs[index]
        except IndexError:
            continue
        soup = parse.to_soup(c)
        dfs[t_id] = parse.get_table(soup, t_id, to_pd=True)

    if output == 'list':
        dfs = list(dfs.values())

    return dfs
