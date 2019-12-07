import pandas as pd


def url_to_id(url: str) -> str:
    """

    """
    return url.split("/")[-1].split(".")[0]


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
    div_dict = {"nhl": div_coords_nhl(div), "nba": div_coords_nba(div)}
    return div_dict[sport]


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
