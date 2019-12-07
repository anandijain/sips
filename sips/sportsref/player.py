"""
used to grab data about players

"""

import os
import pandas as pd

import sips
from sips.h import grab
from sips.h import parse

from sips.macros import sports_ref as sref
from sips.sportsref import utils as sru
from sips.macros import macros as m


divs = {
    "nhl": "div_players",
    "nfl": "div_players",
    "mlb": "div_players_",
}


def player_section_links(sport: str) -> list:
    """

    """
    if sport == "fb":
        prefix = sref.URLS[sport] + "en/players/"
    else:
        prefix = sref.URLS[sport] + "players/"

    # to fix by getting heading links
    if sport == "nfl":
        nfl_letters = sref.LETTERS
        nfl_letters.append("x")
        section_links = [prefix + letter.upper() for letter in nfl_letters]

    elif sport == "fb":
        p = grab.page(prefix)
        index = p.find("ul", {"class": "page_index"})
        a_tags = index.find_all("a")
        section_links = [sref.FB_NS + a_tag["href"] for a_tag in a_tags if a_tag]
    else:
        section_links = [prefix + letter for letter in sref.LETTERS]

    return section_links


def player_links_multi_sports(
    sports: list, concat_dfs: bool = True, write: bool = False, fn: str = "index.csv"
):
    """

    """
    dfs = [player_links(sport, write=write, fn=fn) for sport in sports]

    if concat_dfs:
        ret = pd.concat(dfs)
    else:
        ret = dfs
    return ret


def player_links(
    sport: str, write: bool = False, fn: str = "index.csv"
) -> pd.DataFrame:
    """
    gets the links to every player for a given sport
        - works for mlb, nfl, nhl, and fb
        - not nba

    Args:


    """
    rows = []
    path = m.PARENT_DIR + "data/" + sport + "/players/" + fn
    section_links = player_section_links(sport)
    ps = grab.pages(section_links, output="dict")

    for i, (l, p) in enumerate(ps.items()):

        if sport == "fb":
            div = p.find("div", {"class": "section_content"})
        else:
            div = p.find("div", {"id": divs[sport]})

        if not div:
            continue
        a_tags = div.find_all("a")
        count = 0
        for a_tag in a_tags:
            if not a_tag:
                continue
            link = sref.URLS_ns[sport] + a_tag["href"]
            p_id = sru.url_to_id(link)
            name = a_tag.text
            rows.append([name, p_id, link])
            count += 1
        print(f"{i} : {l} : {count}")

    all_players = pd.DataFrame(rows, columns=["name", "id", "link"])

    if write:
        all_players.to_csv(path)

    return all_players


def player(player_url: str, table_ids: list, output="dict", verbose=False):
    """

    """
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
    """

    """
    path = m.PARENT_DIR + "data/" + sport + "/players/"
    links_df = pd.read_csv(path + "index.csv")

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
    sports = ["nfl", "mlb", "nhl"]
    dfs = [player_links(sport, write=True) for sport in sports]
    print(dfs)

    df = player_links_multi_sports(sports)
    print(df)
