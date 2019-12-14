"""
uses the bovada api to get json data for odds and scores
"""
import requests as r
import pandas as pd

import sips.h.grab as g

from sips.macros import bov as bm
from sips.lines.bov.utils import bov_utils as u


def lines(sports:list, output="df", get_score=True, parse=True, all_mkts=False, verbose=False):
    """
    returns either a dictionary or list
    dictionary - (game_id, row)

    """
    events = u.sports_to_events(sports, all_mkts=all_mkts)

    if output == "dict":
        data = u.dict_from_events(events, key="id", get_score=get_score, rows=parse)

    elif output == "df" or output.lower() == "dataframe":
        data = [u.parse_event(e, get_score=get_score) for e in events]
        if get_score:
            data = pd.DataFrame(data, columns=bm.LINE_COLUMNS)
        else:
            data = pd.DataFrame(data, columns=bm.LINE_COLUMNS_NO_SCORE)
    else:
        data = [u.parse_event(e, get_score=get_score) for e in events]

    if verbose:
        print(f"lines data: {data}")

    return data


def single_game_line(
    sport="basketball/nba",
    a_team="Detroit Pistons",
    h_team="Washington Wizards",
    game_start_time="201911041910",
):
    """
    sport: 3-letter sport
        eg. one in (nfl, nba, nhl)
    teams: [a_team, h_team]
        eg [, ]
    Game date: str Y
        eg.201911041910

    services/sports/event/coupon/events/A/description/
    basketball/nba/
    detroit-pistons-washington-wizards-
    201911041910
    
    """

    a_tm, h_tm = [team.lower().replace(" ", "-") for team in [a_team, h_team]]

    url = (
        bm.BOV_ROOT
        + bm.BOV_EVENT_SFX
        + sport
        + "/"
        + a_tm
        + "-"
        + h_tm
        + "-"
        + game_start_time
    )
    req = g.req_json(url)
    event = req[0]["events"][0]
    row = u.parse_event(event)
    return row


if __name__ == "__main__":
    df = lines(["nba"], output="df", get_score=False, verbose=True)
    # return df
