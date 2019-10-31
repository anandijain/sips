import sips.macros as m
from sips.h import openers as io


def season_boxlinks():
    root = m.sports_ref.fb_url
    ext = "/years/2019/games.htm"

    p = io.get_page(root + ext)
    td_links = p.find_all("td", {"data-stat": "boxscore_word"})

    links = [td_link.a['href'] for td_link in td_links]
    return links
