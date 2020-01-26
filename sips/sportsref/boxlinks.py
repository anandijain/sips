import pandas as pd


from sips.h import grab
from sips.sportsref import utils
from sips.macros import sports_ref as sref


def tag_data_stat(sport: str):
    d = {
        'nhl': {
            'tag': 'th',
            'stat': 'date_game',
            'attr': 'data-stat',
        },
        'nba': {
            'tag': 'td',
            'stat': 'box_score_text',
            'attr': 'data-stat',
        },
        'nfl': {
            'tag': 'td',
            'stat': 'boxscore_word',
            'attr': 'data-stat',
        },
        'mlb': {
            'tag': 'em',
            'stat': None,
            'attr': None,
        },
    }
    return d[sport]['tag'], d[sport]['attr'], d[sport]['stat']


def gen_links(sport: str, start=2019, end=1950):
    sport_caps = sport.upper()
    links = [
        sref.URLS[sport] + f"leagues/{sport_caps}_{year}_games.html"
        for year in range(start, end, -1)
    ]
    return links


def gen_links_mlb(start=2019, end=1910):
    links = [
        sref.URLS['mlb'] + f"leagues/MLB/{year}-schedule.shtml"
        for year in range(start, end, -1)
    ]
    return links


def boxlinks_from_url(link: str, tag_type='td', attr_type=None, data_stat=None):
    boxlinks = []
    page = grab.comments(link)

    if (attr_type is None) and (data_stat is None):
        tags = page.find_all(tag_type)
    else:
        tags = page.find_all(tag_type, {attr_type: data_stat})

    for tag in tags:
        try:
            url = tag.a['href']
            game_id = utils.url_to_id(url)
            boxlinks.append(game_id)
        except AttributeError:
            continue
        except TypeError:
            continue
    return boxlinks  



def sport_boxlinks(sport: str, fn='index.csv', write=False) -> pd.DataFrame:
    all_links = []
    if sport == 'mlb':
        links = gen_links_mlb()
    else:
        links = gen_links(sport)
    
    tag_type, attr, data_stat = tag_data_stat(sport)
    # print(tag_type, attr, data_stat)

    for i, l in enumerate(links):
        boxes = boxlinks_from_url(
            l, tag_type=tag_type, attr_type=attr, data_stat=data_stat)
        print(boxes)

        all_links += boxes
        print(f'{i}: {l} had {len(boxes)} games')

    df = pd.DataFrame(all_links, columns=['game_id'])
    if write:
        folder = utils.gamedata_path(sport)
        df.to_csv(folder + fn)
    return df


if __name__ == "__main__":
    df = sport_boxlinks('mlb', write=True)
