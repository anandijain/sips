import sips.macros as m
import sips.h.grab as g


def season_boxlinks():
    """

    """
    root = m.sports_ref.NFL_URL
    ext = "/years/2019/games.htm"

    p = g.page(root + ext)
    td_links = p.find_all("td", {"data-stat": "boxscore_word"})

    links = [td_link.a["href"] for td_link in td_links]
    return links
