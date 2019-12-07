"""

"""
import pandas as pd
import numpy as np

from sips.macros import sports as sps
from sips.h import helpers as h


def hot_list(strings, output="np"):
    """
    given a list of strings it will return a dict
    string : one hotted np array

    """
    str_set = set(strings)
    length = len(str_set)
    hots = {}
    for i, s in enumerate(str_set):
        hot_arr = np.zeros(length)
        hot_arr[i] = 1
        if hots.get(s) is None:
            if output == "list":
                hot_arr = list(hot_arr)
            hots[s] = hot_arr
    return hots


def hot_teams_dict(sports=["nfl", "nba", "nhl"]):
    """

    """
    team_list = []
    sorted_sports = sorted(sports)
    for s in sorted_sports:
        if s == "nfl":
            team_list += sps.nfl.teams
        elif s == "nba":
            team_list += sps.nba.teams
        elif s == "nhl":
            team_list += sps.nhl.teams
        elif s == "mlb":
            team_list += sps.mlb.teams

    teams_dict = hot_list(team_list, output="list")
    return teams_dict


def hot_statuses_dict():
    statuses = [
        "GAME_END",
        "HALF_TIME",
        "INTERRUPTED",
        "IN_PROGRESS",
        "None",
        "PRE_GAME",
    ]
    statuses_dict = hot_list(statuses, output="list")
    return statuses_dict


def hot_sports_dict():
    statuses = ["BASK", "FOOT", "HCKY", "BASE"]
    statuses_dict = hot_list(statuses, output="list")
    return statuses_dict


def hot_bool_dict(row):
    """
    row type pd.series (row of dataframe)

    """
    hot_mkt = np.array([1, 0]) if row.live else np.array([0, 1])
    return hot_mkt


def all_hot_maps(sports=["nba"], output="dict"):
    """

    """
    hot_maps = [
        hot_sports_dict(),
        hot_teams_dict(sports=sports),
        hot_teams_dict(sports=sports),
        hot_statuses_dict(),
    ]

    if output == "dict":
        keys = ["sport", "a_team", "h_team", "status", "live"]
        hot_maps = {keys[i]: hot_maps[i] for i in range(len(keys) - 1)}

    return hot_maps


def hot(df, hot_maps, drop_cold=True, ret_hots_only=False, verbose=False):
    """
    df: pd.DataFrame
    hot_maps: list(dict)
        hot_map: dict
            key: str column in df
            value: one_hot vector for unique row value
    ---
    returns dataframe 

    """
    if verbose:
        print(f"hot_df cols: {df.columns}")

    ret = []
    for i, (col_name, hot_map) in enumerate(hot_maps.items()):
        ret.append(hot_col(df[col_name], hot_map))
    if ret_hots_only:
        return ret

    ret = pd.concat([df] + ret, axis=1)

    if drop_cold:
        ret = ret.drop(list(hot_maps.keys()), axis=1)

    return ret


def hot_col(col, hot_map, on_keyerror="skip"):
    """
    col: pd.Series
    hot_map: dict
        key: str column in df
        value: one_hot vector for unique row value

    on_keyerror: str 
        one in ['skip', 'set_zero']
    iterates over rows of a column and serializes the 

    """

    hot_cols = list(hot_map.keys())
    hot_dim = len(hot_map[hot_cols[0]])
    hot_rows = []

    for i, elt in col.items():
        try:
            hot_row = hot_map[elt]
        except KeyError:
            if on_keyerror == "skip":
                continue
            elif on_keyerror == "set_zero":
                hot_row = np.zeros(hot_dim)
        hot_rows.append(hot_row)

    hotted_col_df = pd.DataFrame(hot_rows, columns=hot_cols)
    return hotted_col_df


if __name__ == "__main__":
    pass
