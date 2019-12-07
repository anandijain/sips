import pandas as pd

from sips.h import grab
from sips.h import fileio
from sips.sportsref import utils

root = "https://www.basketball-reference.com"
link = root + "/boxscores/shot-chart/201910220TOR.html"


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


def link_to_charts_df(link: str, fn=None):
    """
    df.columns = ['i', 'x', 'y', 'type', 'outcome', 'player', 'game_id']

    """
    game_id = utils.url_to_id(link)
    charts = grab_charts(link=link)
    divs = utils.get_divs(charts)

    if len(divs) == 0:
        return None

    rows = utils.divs_to_arr(divs, "nba")

    df = utils.cat_id(rows, game_id)

    if fn:
        fileio.append_csv(fn, df)

    return df


def all_shotcharts():
    """
    outputs one large DataFrame
    game_id, x, y, type, outcome, player

    """

    write_path = "./data/shots.csv"
    columns = ["i", "x", "y", "type", "outcome", "player", "game_id"]
    fileio.init_csv(write_path, header=columns)

    sfxs = boxlinks()
    meta_df = pd.DataFrame(columns=["game_id", "num_rows"])
    for i, sfx in enumerate(sfxs):
        link = root + sfx
        game_id = utils.url_to_id(sfx)
        df = link_to_charts_df(link, fn=write_path)

        try:
            length = len(df)
        except TypeError:
            continue

        meta_df = meta_df.append(
            {"game_id": game_id, "num_rows": len(df)}, ignore_index=True
        )

        if i % 200 == 0:
            game_id = sfx_to_gameid(sfx)
            print(game_id)
    meta_df.to_csv("./data/meta_shots.csv")


if __name__ == "__main__":
    # main()
    dl = link_to_charts_df(link=link)
    dl
    dl[0]
