import os
import pandas as pd

import sips
from sips.macros import sports_ref as sref
from sips.h import grab
from sips.h import parse
from sips.sportsref.general import player
from sips.sportsref import utils as sru


def player_links():
    all_players = []
    url = sref.fb_url + 'en/players/'   
    p = grab.get_page(url)
    index = p.find('ul', {'class': 'page_index'})
    a_tags = index.find_all('a')
    section_links = [sref.fb_no_slash + a_tag['href'] for a_tag in a_tags if a_tag]
    for i, s in enumerate(section_links):
        print(f'{i}: {s}')
        section_page = grab.get_page(s)
        div = section_page.find('div', {'class': 'section_content'})
        if not div:
            continue
        a_tags = div.find_all('a')
        p_links = [sref.fb_no_slash + a_tag['href'] for a_tag in a_tags if a_tag]
        all_players += p_links

    df = pd.DataFrame(all_players, columns=['link'])
    df.to_csv(sips.PARENT_DIR + 'data/fb/players/index.csv')
    return df



if __name__ == "__main__":
    player_links()
