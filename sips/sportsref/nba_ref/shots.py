import bs4
import re
import pandas as pd

from sips.h import grab
from sips.h import fileio
from sips.sportsref import utils

root = "https://www.basketball-reference.com"
link = root + "/boxscores/shot-chart/"


def grab_charts(link: str):
    """
    given a link to a hockey-reference boxscore, 
    returns div, class: shotchart

    """
    page = grab.page(link)
    charts = page.find_all("div", {"class": "shot-area"})
    return charts


def boxlinks():
    """

    """
    df = pd.read_csv("nba_boxlinks.csv")
    sfxs = df.iloc[:, 1].values
    return sfxs


def div_to_row(div: bs4.BeautifulSoup):
    pos = div.get("style")
    if pos:
        pos.split(";")
    y_pos, x_pos = re.findall(r"\d+", pos)

    tooltip = div.get("class")
    qtr = tooltip[1].split("-")[1]
    p_id = tooltip[2].split("-")[1]
    shot_made = tooltip[3]
    if shot_made == "miss":
        shot_made = 0
    elif shot_made == "make":
        shot_made = 1
    else:
        shot_made = -1  # bad
    tip_str = div.get("tip")
    row = [p_id, qtr, shot_made, x_pos, y_pos, tip_str]
    return row


def link_to_charts_df(link: str) -> dict:
    """
    df.columns = ['i', 'x', 'y', 'type', 'outcome', 'player', 'game_id']

    """
    game_id = utils.url_to_id(link)
    page = grab.page(link)
    dfs = page_to_charts_df(page, game_id)
    return dfs


def page_to_charts_df(page: bs4.BeautifulSoup, game_id: str):
    """

    """
    cols = ["p_id", "qtr", "shot_made", "x_pos", "y_pos", "tip"]
    charts = page.find_all("div", {"class": "shot-area"})

    dfs = {}
    for chart in charts:
        team = chart["id"].split("-")[1]
        divs = chart.find_all("div")
        rows = []
        for div in divs:
            row = div_to_row(div)
            rows.append(row)
        df = pd.DataFrame(rows, columns=cols)
        key = game_id + "_" + team + "_shotchart"
        dfs[key] = df

    return dfs


def all_shotcharts(write=True):
    """
    outputs one large DataFrame
    game_id, x, y, type, outcome, player

    """

    GAMES_DATA = "/home/sippycups/absa/sips/data/nba/games/"
    INDEX_FN = "index.csv"
    df = pd.read_csv(GAMES_DATA + INDEX_FN)
    all_dfs = {}

    for i, game_id in enumerate(df.game_id):
        url = link + game_id + ".html"

        dfs = link_to_charts_df(url)
        num_charts = len(dfs)
        if num_charts == 0:
            print(f"{game_id} had no charts")
            continue
        all_dfs.update(dfs)

        if write:
            for key, val in dfs.items():
                print(key)
                val.to_csv(GAMES_DATA + key + ".csv")

    return all_dfs


if __name__ == "__main__":
    scs = all_shotcharts()
    # print(link)
    # dl = link_to_charts_df(link=link)
    # dl
    # dl[0]
    print(scs.keys())
    # print(dl)
