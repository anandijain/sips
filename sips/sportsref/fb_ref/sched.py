import sips.sportsref.h as ref_utils


def get_sched():
    root = ref_utils.macros.fb_url
    ext = "/years/2019/games.htm"

    p = ref_utils.openers.page(root + ext)
    td_links = p.find_all("td", {"data-stat" : "boxscore_word"})

    links = [td_link.a['href'] for td_link in td_links]
    return links
    
