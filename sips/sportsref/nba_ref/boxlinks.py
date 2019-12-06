"""
get boxlinks

"""

import sips.h.grab as g
import sips.h.parse as p


def get_links_to_months(page):
    """
    gets the filter header that links to the boxlinks for the season
    returns the links
    """
    d = page.find("div", {"class": "filter"})
    atags = d.find_all("a")
    links_to_months = [atag["href"] for atag in atags]
    return links_to_months


def month_of_boxlinks(link):
    """
    given a link to a particular month in a season schedule 
    returns a list of the games' boxlinks 
    """
    t = p.get_table(g.page(link), "schedule")
    columns = p.columns_from_table(t, attr="data-stat")
    trs = t.tbody.find_all("tr")
    ths = [tr.th for tr in trs]
    boxlinks = [th.a["href"] for th in ths]
    return boxlinks


def main():
    """

    """
    root = "https://www.basketball-reference.com"
    url = "/leagues/NBA_2020_games.html"

    page = g.page(root + url)

    links_to_months = get_links_to_months(page)
    all_boxlinks = []
    for mlink in links_to_months:
        all_boxlinks += month_of_boxlinks(root + mlink)
    return all_boxlinks


if __name__ == "__main__":
    all_boxlinks = main()
    print(f"all_boxlinks: {all_boxlinks}")
