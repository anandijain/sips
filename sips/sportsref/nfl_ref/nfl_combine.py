import pandas as pd

import sips.h.parse as p
import sips.h.grab as g


def main(years=(2000, 2019)):
    """
    + 1 because range is not inclusve

    """
    year_list = range(years[0], years[1] + 1)
    dfs = []
    for year in year_list:
        dfs.append(get_df(year))
    print(f"Done: {len(dfs)} dataframes written")


def get_df(year, write=True):
    """

    """
    url = get_url(year=year)
    page = g.page(url)
    table = p.get_table(page, "combine")
    cols = p.columns_from_table(table)
    player_ids = get_ids(table)
    raw_df = pd.read_html(table.prettify())[0]
    df = cat_ids(raw_df, player_ids)
    print(df)
    if write:
        fn = get_fn(year)
        df.to_csv(fn)
    return df


def get_url(year=2019):
    """

    """
    root = "https://www.pro-football-reference.com"
    url = "/draft/" + str(year) + "-combine.htm"
    return root + url


def get_ids(table):
    """

    """
    ids = []
    players = table.tbody.find_all("th", {"data-stat": "player"})
    for player in players:
        try:
            player_url = player.a["href"]
            player_id = parse_id(player_url)
            ids.append(player_id)
        except TypeError:
            ids.append(player.text)
    return ids


def parse_id(player_url="/players/W/WoodZe00.htm"):
    """

    """
    ID = player_url.split("/")[3].split(".")[0]
    return ID


def cat_ids(raw_df, list_ids):
    """

    """
    id_col = pd.Series(list_ids, name="id")
    df = pd.concat([raw_df, id_col], axis=1)
    return df


def get_fn(year):
    return "./data/" + str(year) + "_nfl_combine.csv"


if __name__ == "__main__":
    main()
