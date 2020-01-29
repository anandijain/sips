import glob
import numpy as np
import pandas as pd

from sips.macros import macros as m


def group_read(table_type: str, sport: str, apply:list=[]):
    dfs = {}
    path_to_data = gamedata_path(sport)

    fns = glob.glob(path_to_data + f'**{table_type}.csv')
    print(f'reading {len(fns)} files')
    for i, fn in enumerate(fns):
        try:
            df = pd.read_csv(fn)
        except:
            continue

        for fxn in apply:
            df = fxn(df)

        game_id = path_to_id(fn)
        df['game_id'] = game_id
        dfs[game_id] = df
    return dfs


def split_str_times_df(df: pd.DataFrame, col='time', out_cols=['mins', 'secs'], drop_old=True):
    times = df[col]
    ms = times.str.split(':', expand=True).astype(np.float)
    df['mins'] = ms[0]
    df['secs'] = ms[1]
    if drop_old:
        df = df.drop(col, axis=1)
    return df


def drop_rename(df: pd.DataFrame, columns, drop_n=0):
    """

    """
    df = df.drop(range(drop_n))
    df.columns = columns
    if "index" in df.columns:
        df = df.drop("index", axis=1)
    return df


def drop_rename_from_fn(fn: str, cols, id_col='game_id', drop_n=0, verbose=False):
    """

    """
    df = pd.read_csv(fn)
    if verbose:
        print(f'pre: {df}')
    df = drop_rename(df, cols, drop_n=drop_n)
    df = add_id_from_fn(df, fn, id_col)
    return df


def add_id_from_fn(df: pd.DataFrame, fn: str, col='player_id'):
    """
    given a dataframe and a filename
    adds a id col to the dataframe for the id obtained from filename
    """
    obj_id = path_to_id(fn)  # player or game
    df[col] = obj_id
    return df


def drop_ith_col(fn: str, i):
    # drop a col by index
    df = pd.read_csv(fn)
    df = df.drop(df.columns[i], axis=1)
    return df


def game_id_to_home_code(game_id: str):
    return game_id[-3:]


def mlb_game_id_to_home_code(game_id: str):
    return game_id[:3]


def url_to_id(url: str) -> str:
    """

    """
    return url.split("/")[-1].split(".")[0]


def path_to_id(path:str) -> str:
    return url_to_id(path).split('_')[0]


def gamedata_path(sport: str, cloud=False):
    if cloud:
        path = ''
    else:
        path = m.PARENT_DIR + 'data/' + sport + '/games/'
    return path


def id_to_sfx(id: str) -> str:
    return


def get_divs(charts):
    """

    """
    all_divs = []
    for chart in charts:
        divs = chart.find_all("div")
        all_divs += divs
    return all_divs


def divs_to_arr(divs, sport):
    """

    """
    rows = []
    for div in divs:
        try:
            dict = arr_row(div, sport)
        except KeyError:
            continue
        rows.append(dict)
    return rows


def arr_row(div, sport: str):
    """
    game_id, x, y, shot_type, title, player

    """
    x, y = div_coords(div, sport)
    type = shot_type(div["class"])
    title, player = shot_title(div["title"])
    return [x, y, type, title, player]


def div_coords(div, sport: str):
    if sport == "nhl":
        div_dict = div_coords_nhl(div)
    elif sport == "nba":
        div_dict = div_coords_nba(div)
    return div_dict


def div_coords_nhl(div):
    # positions = top == y, left == x
    positions = div["style"].split(" ")
    x, y = positions[3], positions[1]
    x, y = [int(c.split("p")[0]) for c in (x, y)]
    return x, y


def div_coords_nba(div):
    """
    positions = top == y, left == x

    """
    print(f"div: {div}")
    positions = div["style"].split(";")
    pos = positions
    x, y = pos[1], pos[0]
    x, y = [int(c.split(":")[1].split("p")[0]) for c in (x, y)]
    return x, y


def cat_id(rows: list, id):
    """

    """
    df = pd.DataFrame(rows)
    df["game_id"] = id
    return df


def shot_title(title):
    """

    """
    shot_outcome, player = title.split(" - ")
    return shot_outcome, player


def shot_type(t):
    """

    """
    ret = None
    if len(t) > 1:
        t = " ".join(t)
        ret = t
    else:
        ret = t[0]
    return ret


def div_dict_row(div, dict, sport):
    """

    """
    x, y = div_coords(div, sport)
    type = shot_type(div["class"])
    title, player = shot_title(div["title"])

    dict["x"].append(x)
    dict["y"].append(y)
    dict["shot_type"].append(type)
    dict["title"].append(title)
    dict["player"].append(player)

    return dict


def parse_chart(divs, game_id, sport):
    """
    game_id, x, y, shot_type, title, player

    """
    print(divs)
    print(len(divs))
    ids = [game_id for _ in range(len(divs["x"]))]
    dict = {
        "game_id": ids,
        "x": [],
        "y": [],
        "shot_type": [],
        "title": [],
        "player": [],
    }
    cs = None
    for div in divs:
        try:
            cs = div_coords(div, sport)
        except KeyError:
            continue
        dict = div_dict_row(div, dict, sport)

    return dict
