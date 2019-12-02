import os
import pandas as pd

import sips
from sips.macros import sports_ref as sref
from sips.sportsref import utils as sru
from sips.h import grab
from sips.h import parse

divs = {
    'nhl': "div_players",
    'nfl': "div_players",
    'mlb': "div_players_",

}

def player_section_links(sport:str) -> list:
    if sport == 'fb':
        prefix = sref.urls[sport] + "en/players/"
    else:
        prefix = sref.urls[sport] + "players/"

    # to fix by getting heading links
    if sport == 'nfl':
        nfl_letters = sref.letters
        nfl_letters.append('x')
        section_links = [prefix + letter.upper() for letter in nfl_letters]
    elif sport == 'fb':
        p = grab.page(prefix)
        index = p.find("ul", {"class": "page_index"})
        a_tags = index.find_all("a")
        section_links = [sref.fb_ns + a_tag["href"] for a_tag in a_tags if a_tag]
    else:
        section_links = [prefix + letter for letter in sref.letters]

    return section_links


def player_links(sport: str, write: bool = False, fn: str='index.csv') -> pd.DataFrame:
    """
    gets the links to every player for a given sport
        - works for mlb, nfl, nhl, and fb
        - not nba

    """
    all_players = []
    path = sips.PARENT_DIR + "data/" + sport + "/players/" + fn
    section_links = player_section_links(sport)
    ps = grab.pages(section_links, output='dict')

    for i, (l, p) in enumerate(ps.items()):

        if sport == 'fb':
            div = p.find("div", {"class": "section_content"})
        else:
            div = p.find("div", {"id": divs[sport]})

        if not div:
            continue
        a_tags = div.find_all("a")
        p_links = [sref.urls_ns[sport] + a_tag["href"]
                   for a_tag in a_tags if a_tag]
        all_players += p_links
        
        print(f"{i} : {l} : {len(p_links)}")

    df = pd.DataFrame(all_players, columns=["link"])

    if write:
        df.to_csv(path)

    return df


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
