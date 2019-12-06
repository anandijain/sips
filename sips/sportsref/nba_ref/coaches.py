import pandas as pd

import sips
from sips.h import grab
from sips.h import parse
from sips.macros import sports_ref as sref


def coaches():
    """

    """
    p = sips.page(sref.NBA_URL + "/coaches/")
    t = p.find("table", {"id": "coaches"})
    ctags = t.find_all("th", {"data-stat": "coach"})
    links = []
    for c in ctags:
        link = c.find("a")
        if link:
            links.append(sref.NBA_URL + link["href"])

    return links


def coach_stats():
    """

    """
    links = coaches()
    tables = grab.tables_from_links(links, ["coach-stats"])
    return tables


# jank cuz comments
def coach_awards():
    """

    """
    links = coaches()
    pages = grab.pages(links)
    dfs = []
    for p in pages:
        t = parse.comments(p)[28]  # hack
        df = pd.read_html(t)
        dfs.append(df)
    return dfs


if __name__ == "__main__":
    dfs = coach_stats()
    print(dfs)

    dfs = coach_awards()
    print(dfs)
