"""
get boxlinks

"""
import pandas as pd

import sips.h.grab as g
import sips.h.parse as p
from sips.sportsref import utils

nba_months = [
    "october",
    "november",
    "december",
    "january",
    "february",
    "march",
    "april",
]


def get_links_to_months(page):
    """
    gets the filter header that links to the boxlinks for the season
    returns the links
    """
    div = page.find("div", {"class": "filter"})
    atags = div.find_all("a")
    links_to_months = [atag["href"] for atag in atags]
    return links_to_months


def month_of_boxlinks(link):
    """
    given a link to a particular month in a season schedule 
    returns a list of the games' boxlinks 
    """
    boxlinks = []
    t = p.get_table(g.page(link), "schedule")
    # columns = p.columns_from_table(t, attr="data-stat")
    trs = t.tbody.find_all("tr")
    ths = [tr.th for tr in trs]
    for th in ths:
        a_tag = th.get("a")
        if a_tag:
            url = a_tag.get("href")
            if url is not None:
                boxlinks.append(url)

    return boxlinks


def all_nba_boxlinks(to_pd=False):
    """

    """
    root = "https://www.basketball-reference.com"
    all_boxlinks = []
    links = [
        root + "/leagues/NBA_" + str(year) + "_games-" + month + ".html"
        for year in range(2020, 1950, -1)
        for month in nba_months
    ]
    print(links)
    for i, link in enumerate(links):
        page = g.page(link)
        boxlinks = page.find_all("td", {"data-stat": "box_score_text"})
        # print(boxlinks)
        for bl in boxlinks:
            a_tag = bl.a
            if a_tag is not None:
                game_id = utils.url_to_id(a_tag["href"])
                # print(game_id)
                all_boxlinks.append(game_id)
        if i % 5 == 0:
            print(f"{i}: {link}")

    if to_pd:
        all_boxlinks = pd.DataFrame(all_boxlinks, columns=["game_id"])
        all_boxlinks.to_csv("nba_boxlinks.csv")

    return all_boxlinks


if __name__ == "__main__":
    all_boxlinks = all_nba_boxlinks()
    print(f"all_boxlinks: {all_boxlinks}")
