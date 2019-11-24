import pandas as pd
import numpy as np

from sips.macros import sports
import sips.h.helpers as h


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


def hot_teams_dict(teams_to_hot=["nfl", "nba", "nhl"]):
    """
    order matters,
    """
    team_list = []
    sorted_sports = sorted(teams_to_hot)
    for s in sorted_sports:
        if s == "nfl":
            team_list += sports.nfl.teams
        elif s == "nba":
            team_list += sports.nba.teams
        elif s == "nhl":
            team_list += sports.nhl.teams
        elif s == "mlb":
            team_list += sports.mlb.teams

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


def dicts_for_one_hotting(sports=["nfl", "nba", "nhl"]):
    teams_dict = hot_teams_dict(teams_to_hot=sports)
    statuses_dict = hot_statuses_dict()
    return teams_dict, statuses_dict


def all_hot_maps(output="dict"):
    hot_maps = [
        hot_sports_dict(),
        hot_teams_dict(),
        hot_teams_dict(),
        hot_statuses_dict(),
    ]

    if output == "dict":
        keys = ["sport", "a_team", "h_team", "status", "live"]
        hot_maps = {keys[i]: hot_maps[i] for i in range(len(keys) - 1)}

    return hot_maps


def hot(df, hot_maps, drop_cold=True, ret_hots_only=False):
    """
    df: pd.DataFrame
    hot_maps: list(dict)
        hot_map: dict
            key: str column in df
            value: one_hot vector for unique row value
    ---
    returns dataframe 

    """
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


def freeze_(df,):
    """

    df: pd.DataFrame

    """
    pass


def freeze(df: pd.DataFrame, cold_map: list):
    """
    df: pd.DataFrame
    cold_map: list(str) where len(cold_map) == df.shape[1]
    ---

    for a series with n unique values, the hot vector will be n columns 
        ## - how to deal with set_zero?
    
    freeze assumes that there is this bijection from hot values to cold keys
    """
    pass


def test_phase_changes():
    dfs = h.get_dfs()
    df = dfs[0]

    hot_maps = all_hot_maps(output="dict")
    hotted = hot(df, hot_maps=hot_maps)
    return hotted


if __name__ == "__main__":
    hots = test_phase_changes()
    print(hots)
