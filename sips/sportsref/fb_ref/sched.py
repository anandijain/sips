import sips.h as h


def season_boxlinks():
    root = h.new_macros.sports_ref.fb_url
    ext = "/years/2019/games.htm"

    p = h.openers.get_page(root + ext)
    td_links = p.find_all("td", {"data-stat": "boxscore_word"})

    links = [td_link.a['href'] for td_link in td_links]
    return links
