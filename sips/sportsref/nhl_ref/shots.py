import bs4
import requests as r

import pandas as pd
import numpy as np

import sips.h.parse as parse
from sips.h import fileio

from sips.sportsref import utils


link = "https://www.hockey-reference.com/boxscores/201904100NYI.html"
root = "https://www.hockey-reference.com"


def grab(link, fn=None):
    """

    """
    game_id = utils.url_to_id(link)
    charts = grab_charts(link=link)
    divs = utils.get_divs(charts)

    if len(divs) == 0:
        return None

    rows = utils.divs_to_arr(divs, "nhl")

    df = utils.cat_id(rows, game_id)
    # df.columns = ['i', 'x', 'y', 'type', 'outcome', 'player', 'game_id']

    if fn:
        fileio.append_csv(fn, df)

    return df


def grab_charts(link):
    """
    given a link to a hockey-refference boxscore, 
    returns div, class: shotchart

    """
    req = r.get(link).text
    p = bs4.BeautifulSoup(req, "html.parser")
    cs = parse.comments(p)
    shotchart_comment = cs[22]
    chart_html = bs4.BeautifulSoup(shotchart_comment, "html.parser")
    charts = chart_html.find_all("div", {"class": "shotchart"})
    return charts


def boxlinks():
    df = pd.read_csv("nhl_boxlinks.csv")
    sfxs = df.iloc[:, 1].values
    return sfxs


def main():
    """
    outputs one large DataFrame
    game_id, x, y, type, outcome, player

    """

    write_path = "./data/shots.csv"
    columns = ["i", "x", "y", "type", "outcome", "player", "game_id"]

    f = fileio.init_csv(write_path, header=columns)

    sfxs = np.random.permutation(boxlinks())
    meta_df = pd.DataFrame(columns=["game_id", "num_rows"])
    for i, sfx in enumerate(sfxs):
        link = root + sfx
        game_id = utils.url_to_id(sfx)
        df = grab(link, fn=write_path)
        try:
            length = len(df)
        except TypeError:
            continue

        meta_df = meta_df.append(
            {"game_id": game_id, "num_rows": len(df)}, ignore_index=True
        )

        if i % 200 == 0:
            game_id = utils.url_to_id(sfx)
            print(game_id)
    meta_df.to_csv("./data/meta_shots.csv")


if __name__ == "__main__":
    main()
