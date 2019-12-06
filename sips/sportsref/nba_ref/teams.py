import pandas as pd

from sips.macros import sports_ref as sref
from sips.h import grab
from sips.h import parse


def get_links():
    """

    """
    t = grab.get_table(sref.NBA_URL + "/teams/", ["teams_active"], to_pd=False)
    links = [sref.NBA_URL + link["href"] for link in t.find_all("a")]
    return links


def get_teams(output="list"):
    """

    """
    team_links = get_links()
    team_pages = grab.pages(team_links, output=output)
    return team_pages


def get_histories():
    """

    """
    team_histories = []
    teams_dict = get_teams(output="dict")  # link : page

    for link, page in teams_dict.items():
        team_name = link.split("/")[-1]
        print(link)
        print(f"team_name: {team_name}")
        df = grab.get_table(link, [team_name])
        team_histories.append(df)
    return team_histories


if __name__ == "__main__":
    dfs = get_histories()
    print(dfs)
