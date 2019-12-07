import pandas as pd
from sips.h import grab as g

url = "https://www.hockey-reference.com"


def league_index():
    """

    """
    suffix = "/leagues/"
    p = g.page(url + suffix)
    t = p.find("table", {"id": "league_index"})
    return t


def find_in_table(t, tup):
    """
    tup is 3tuple with ('th', 'data-stat', 'season') for example

    """
    selected = t.find_all(tup[0], {tup[1]: tup[2]})
    links = []
    for sel in selected:
        try:
            links.append(sel.a["href"])
        except TypeError:
            continue
    return links


def link_fix(link):
    """

    """
    split = link.split(".")
    split[0] += "_games."
    ret = split[0] + split[1]
    return ret


def gamelinks_str_fix(links):
    """

    """
    ret = []
    for link in links:
        ret.append(link_fix(link))
    return ret


def season_boxlinks(season_url):
    """

    """
    find_tup = ("th", "data-stat", "date_game")
    p = g.page(season_url)
    tables = p.find_all("table")
    ret = []
    for table in tables:
        ret += find_in_table(table, find_tup)
    return ret


def parse_box(boxlink):
    """

    """
    p = g.page(boxlink)
    tables = p.find_all("table")
    dfs = []
    for table in tables:
        ids = find_in_table(table, ("td", "data-stat", "player"))
        if not len(ids):
            continue
        print(ids)
        df = pd.read_html(table.prettify())[0]
        df.columns = df.columns.droplevel()
        id_series = pd.Series(ids, name="id", dtype="object")
        df["id"] = id_series
        dfs.append(df)
    return dfs


def main(write=True):
    """

    """
    ret = []
    t = league_index()
    ls = find_in_table(t, ("th", "data-stat", "season"))
    ls = gamelinks_str_fix(ls)
    for l in ls[1:]:
        print(l)
        ret += season_boxlinks(url + l)
    if write:
        df = pd.Series(ret, name="boxlinks")
        df.to_csv("nhl_boxlinks.csv")
    return ret
