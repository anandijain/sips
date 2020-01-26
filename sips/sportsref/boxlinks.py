from sips.h import grab
from sips.sportsref import utils


def boxlinks_from_table(link: str, data_stat='box_score_text'):
    boxlinks = []
    page = grab.page(link)
    tds = page.find_all("td", {"data-stat": data_stat})
    # print(boxlinks)
    for td in tds:
        try:
            url = td.a['href']
            game_id = utils.url_to_id(url)
            boxlinks.append(game_id)
        except AttributeError:
            continue
        except TypeError:
            continue
    return boxlinks
