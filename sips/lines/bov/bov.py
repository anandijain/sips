"""
uses the bovada api to get json data for odds and scores
"""
import requests as r
import pandas as pd
import numpy as np

import sips.h.grab as g
import sips.h.serialize as s
import sips.h.helpers as h
import sips.h.analyze as analyze

from sips.macros import bov as bm
from sips.lines.bov.utils import bov_utils as u


def lines(sports, output="list", parse=True, all_mkts=False, verbose=False):
    """
    returns either a dictionary or list
    dictionary - (game_id, row)
    """
    events = u.sports_to_events(sports, all_mkts)

    if output == "dict":
        data = u.dict_from_events(events, key="id", rows=parse)
    else:
        data = [u.parse_event(e) for e in events]

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
    print(url)
    req = g.req_json(url)
    print(req)
    event = req[0]["events"][0]
    row = u.parse_event(event)
    return row


def prep_game_dataset(fn, sports=["nba"]):  # , zip_data=True, verbose=False):
    teams_dict, statuses_dict = h.dicts_for_one_hotting()

    df = pd.read_csv(fn)
    prev = [None, None]
    prev_row = [None for _ in range(25)]
    X = []
    y = []
    for i, row in df.iterrows():

        cur_row = row.values
        cur_ml = list(row[["a_ml", "h_ml"]])
        if i == 0:
            prev_ml = cur_ml
            prev_row = cur_row
            continue
        transition_class = analyze.classify_transition(prev_ml, cur_ml)
        if bm.TRANSITION_CLASS_STRINGS[np.argmax(transition_class)] == "stays same":
            continue

        x = s.serialize_row(prev_row, teams_dict, statuses_dict)
        y.append(transition_class)
        X.append(x)
        prev_ml = cur_ml
        prev_row = cur_row
    # ret = [X, y]

    # if zip_data:
    #     ret = list(zip(X, y))

    # if verbose:
    #     print(f'game dataset: {ret}')
    len_game = len(y)
    if not X:
        return
    X = np.reshape(np.concatenate(X, axis=0), (len_game, 1, -1))
    # y = np.reshape(np.concatenate(y, axis=0), (466, 1, -1))

    return X, y


def main():
    # data = lines(["nba"], output='dict')
    # print(data)
    # print(len(data))
    # return data
    row = single_game_line()
    print(row)
    return row


if __name__ == "__main__":
    data = main()
