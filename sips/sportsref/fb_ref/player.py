import os
import pandas as pd

import sips
from sips.macros import sports_ref as sref
from sips.h import grab
from sips.h import parse
from sips.sportsref import utils as sru


def player_links():
    all_players = []
    url = sref.fb_url + "en/players/"
    p = grab.get_page(url)
    index = p.find("ul", {"class": "page_index"})
    a_tags = index.find_all("a")
    section_links = [sref.fb_no_slash + a_tag["href"]
                     for a_tag in a_tags if a_tag]
    for i, s in enumerate(section_links):
        print(f"{i}: {s}")
        section_page = grab.get_page(s)
        div = section_page.find("div", {"class": "section_content"})
        if not div:
            continue
        a_tags = div.find_all("a")
        p_links = [sref.fb_no_slash + a_tag["href"]
                   for a_tag in a_tags if a_tag]
        all_players += p_links

    df = pd.DataFrame(all_players, columns=["link"])
    df.to_csv(sips.PARENT_DIR + "data/nfl/players/index.csv")
    return df


def player(url):
    p = grab.get_page(url)
    prefix = 'stats_'
    sfxs = ["_ks_dom_lg", "_ks_dom_cup",
            "_ks_intl_cup", "_ks_expanded", "_ks_collapsed"]
    categories = ['standard', 'shooting', 'passing', 'playing_time', 'misc']
    table_ids = []
    for cat in categories:
         table_ids += [prefix + cat + sfx for sfx in sfxs]
    
    table_ids.append(prefix + 'player_summary')

    p_dfs = {}

    for t_id in table_ids:
        df = parse.get_table(p, t_id, to_pd=True)
        if df is None:
            continue
        p_dfs[t_id] = df
    return p_dfs


def main():
    path = sips.PARENT_DIR + 'data/fb/players/'
    links_path = path + 'index.csv'
    players_df = pd.read_csv(links_path)
    links = players_df.link

    for i, link in enumerate(links):
        
        p_dfs = player(link)
        p_id = sru.url_to_id(link)
        
        split_url = link.split("/")
        p_folder = split_url[-1] + '_' + split_url[-2]
        player_dir = path + p_folder + '/'

        print(f'{i}: {p_folder}')

        if not os.path.isdir(player_dir):
            os.mkdir(player_dir)
            
        if p_dfs is None:
            continue

        for t_id, df in p_dfs.items():
            fn = player_dir + t_id + '.csv'
            df.to_csv(fn)

if __name__ == "__main__":
    p_dfs = player('https://fbref.com/en/players/d70ce98e/Lionel-Messi')
    print(p_dfs)
