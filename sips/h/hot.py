import pandas as pd
import numpy as np

from sips.macros import sports


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
    '''
    order matters,
    '''
    team_list = []
    sorted_sports = sorted(teams_to_hot)
    for s in sorted_sports:
        if s == "nfl":
            team_list += sports.nfl.teams
        elif s == "nba":
            team_list += sports.nba.teams
        elif s == "nhl":
            team_list += sports.nhl.teams
        elif s == 'mlb':
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
    statuses = [
        "BASK",
        "FOOT",
        "HCKY",
        "BASE",
        "None",
    ]
    statuses_dict = hot_list(statuses, output="list")
    return statuses_dict


def dicts_for_one_hotting(sports=["nfl", "nba", "nhl"]):
    teams_dict = hot_teams_dict(teams_to_hot=sports)
    statuses_dict = hot_statuses_dict()
    return teams_dict, statuses_dict


def hot(df, cols_to_hot=['sport', 'a_team', 'h_team', 'status'], hot_maps=[hot_sports_dict(), hot_teams_dict(), hot_teams_dict(), hot_statuses_dict()]):
    '''
    let m == len(hot_maps)

    m == len(columns)
    or hot_maps == None
    first try:
        - grab columns to hot
        - create m many dfs of the hotted data 
        - concat onto df
    '''
    hot_dfs = []
    hot_tups = []
    for i, col in enumerate(cols_to_hot):
        if col in df.columns:
            hot_tups.append((df[col], hot_maps[i]))

    for col, hot_map in hot_tups:
        hot_dfs.append(hot_col(col, hot_map))
    ret = pd.concat([df] + hot_dfs, axis=1)
    ret = ret.drop(cols_to_hot, axis=1, errors='ignore')
    return ret


def hot_col(col, hot_map):
    """

    """
    hot_cols = hot_map.keys()
    hot_rows = []
    for i, elt in col.items():

        hot_rows.append(hot_map[elt])

    hotted_col_df = pd.DataFrame(hot_rows, columns=hot_cols)
    return hotted_col_df
