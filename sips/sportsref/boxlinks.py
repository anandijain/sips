import pandas as pd


from sips.h import grab
from sips.sportsref import utils
from sips.macros import sports_ref as sref


def gen_links(sport: str, start=2019, end=1950):
    sport_caps = sport.upper()
    links = [
        sref.URLS[sport] + f"leagues/{sport_caps}_{year}_games.html"
        for year in range(start, end, -1)
    ]
    return links


def boxlinks_from_url(link: str, data_stat, tag_type='td'):
    boxlinks = []
    page = grab.comments(link)
    tags = page.find_all(tag_type, {"data-stat": data_stat})
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


def tag_data_stat(sport:str):
    d = {
        'nhl': {
            'tag' : 'th',
            'stat' : 'date_game',
        },
        'nba': {
            'tag' : 'td',
            'stat' : 'box_score_text',
        },
        'nfl': {
            'tag' : 'td',
            'stat' : 'boxscore_word',
        },
    }
    return d[sport]['tag'], d[sport]['stat']


def sport_boxlinks(sport:str, fn='index.csv', write=False) -> pd.DataFrame:
    all_links = []
    links = gen_links(sport)
    tag_type, data_stat = tag_data_stat(sport)

    for i, l in enumerate(links):
        boxes = boxlinks_from_url(l, data_stat, tag_type=tag_type)
        all_links += boxes
        print(f'{i}: {l} had {len(boxes)} games')

    df = pd.DataFrame(all_links, columns=['game_id'])
    if write:
        folder = utils.gamedata_path(sport)
        df.to_csv(folder + fn)
    return df


if __name__ == "__main__":
    df = sport_boxlinks('nhl', write=True)
