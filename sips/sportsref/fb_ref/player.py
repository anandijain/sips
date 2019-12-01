import os
import time
import pandas as pd

import sips
from sips.macros import sports_ref as sref
from sips.h import grab
from sips.h import parse
from sips.sportsref.general import player
from sips.sportsref import utils as sru

comment_idxs = {
    20: 'detailed_rushing_and_receiving',
    24: 'returns',
    25: 'defense',
    29: 'scoring',
    30: 'snap_counts',
    33: 'combine',
}


def player_links(output='df'):
    letters = ['a', 'b', 'c', "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
               "o",  "p", "q", "r", "s", "t", "u", 'v', 'w', 'x', 'y', 'z']
    all_links = []

    section_links = [sref.fb_url + 'players/' + letter.upper()
                     for letter in letters]

    ps = {l: grab.get_page(l) for l in section_links}
    for l, p in ps.items():
        div = p.find('div', {'id': 'div_players'})
        if not div:
            print(l)
            continue
        a_tags = div.find_all('a')
        links = [a['href'] for a in a_tags]
        all_links += links
    if output == 'df':
        all_links = pd.DataFrame(all_links, columns=['link'])
        # all_links.to_csv()
    return all_links


if __name__ == "__main__":
    # ls = player_links()
    table_ids = ['stats', 'rushing_and_receiving']

    path = sips.PARENT_DIR + 'data/nfl/players/'
    player_links_path = path + 'index.csv'
    df = pd.read_csv(player_links_path)

    ps = {}
    for i, link in enumerate(df.link):
        player_url = sref.fb_no_slash + link
        p_id = sru.url_to_id(player_url)
        
        player_path = path + p_id + '/'
        print(f'{i}: {player_url}')

        if not os.path.isdir(player_path):
            os.mkdir(player_path)

        p = player.player(player_url, table_ids, comment_idxs)
        ps[link] = p
        time.sleep(0.15)
        for t_id, df in p.items():
            if not df:
                continue
            df = df[0]
            fn = p_id + '_' + t_id
            df.to_csv(player_path + fn + '.csv')

    print(ps)
