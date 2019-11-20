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


def hot_statuses():
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


def dicts_for_one_hotting(sports=["nfl", "nba", "nhl"]):
    teams_dict = hot_teams_dict(teams_to_hot=sports)
    statuses_dict = hot_statuses()
    return teams_dict, statuses_dict


def hot(df, columns=['status'], hot_maps=None):
    '''
    let m == len(hot_maps)

    m == len(columns)
    or hot_maps == None
    first try:
        - grab columns to hot
        - create m many dfs of the hotted data 
        - concat onto df
    '''
    if not hot_maps:
        return pd.get_dummies(df, columns=columns)

    to_hot = df[columns]
    hot_dfs = []
    for col, hot_map in zip(to_hot.iteritems(), hot_maps):
        hot_dfs.append(hot_col(col, hot_map))
    ret = pd.concat([df] + hot_dfs, axis=1)
    ret = ret.drop(columns, axis=1)
    return ret


def hot_col(col, hot_map):
    hot_cols = hot_map.keys()
    hot_rows = []
    for i, elt in col[1].items():
        hot_rows.append(hot_map[elt])
    hotted_col = pd.DataFrame(hot_rows, columns=hot_cols)
    return hotted_col
