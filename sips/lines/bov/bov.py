'''
uses the bovada api to get json data for odds and scores
'''
import requests as r

import numpy as np

import sips.h.grab as g

from sips.macros import bov as bm
from sips.lines.bov.utils import bov_utils as u


def lines(sports, output='list', parse=True, all_mkts=False, verbose=False):
    '''
    returns either a dictionary or list
    dictionary - (game_id, row)
    '''
    events = u.sports_to_events(sports, all_mkts)

    if output == 'dict':
        data = u.dict_from_events(events, key='id', rows=parse)
    else:
        data = [u.parse_event(e) for e in events]

    return data


def single_game_line(sport='basketball/nba', a_team='Detroit Pistons', h_team='Washington Wizards', game_start_time='201911041910'):
    '''
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
    '''

    a_tm, h_tm = [team.lower().replace(' ', '-') for team in [a_team, h_team]]

    url = bm.BOV_ROOT + bm.BOV_EVENT_SFX + sport + '/' + \
        a_tm + '-' + h_tm + '-' + game_start_time
    print(url)
    req = g.req_json(url)
    print(req)
    event = req[0]['events'][0]
    row = u.parse_event(event)
    return row


def serialize_row(row, teams_dict, statuses_dict):
    '''
    going to take in something like this:
    ['FOOT', 5741304, 'Pittsburgh Steelers', 'Cleveland Browns', 1573540736617, 28, 
    False, '0', '-1', '0', '0', 'PRE_GAME', '2.5', '-2.5', '-105', '-115', '+125', 
    '-145', '40.0', '40.0', '-110', '-110', 'O', 'U', 1573780800000]
    and return a np array 
    '''
    ret = []

    teams = row[2:4]


    for t in teams:
        ret += teams_dict[t]
    
    ret += row[4:6]

    if row[6]:
        ret += [1, 0]
    else:
        ret += [0, 1]

    ret += row[7:11]

    row_status = row[11]
    hot_status = statuses_dict[row_status]
    ret += hot_status

    mls = [100 if ml == 'EVEN' else ml for ml in row[12:22]]
    final = np.array(ret, dtype=np.float32)
    return final


def classify_transition(prev_mls, cur_mls):
    '''
    uses the propositions described in transitions() to return a numpy array
    with the class of transition corresponding to the input moneylines
    '''
    a_prev = prev_mls[0]
    a_cur = cur_mls[0]

    h_prev = prev_mls[1]
    h_cur = cur_mls[1]

    propositions = transitions(a_prev, a_cur, h_prev, h_cur)
    ret = np.zeros(len(propositions))

    for i, phi in enumerate(propositions):
        if phi:
            ret[i] = 1
            break

    return ret


def transitions(a1, a2, h1, h2):
    '''
    classification of the movement of lines where -1 is closed
    '''
    # how to metaprogram the enumeration of combinations given binary relations
    propositions = [
        # opening actions
        ((a1 == -1 and a2 != -1) and (h1 == -1 and h2 != -1)),
        ((a1 == -1 and a2 != -1) and (h1 < h2)),
        ((a1 == -1 and a2 != -1) and (h1 > h2)),
        ((a1 < a2) and (h1 == -1 and h2 != -1)),
        ((a1 > a2) and (h1 == -1 and h2 != -1)),
        # closing actions
        ((a1 and a2 == -1) and (h1 and h2 == -1)),
        ((a1 and a2 == -1) and (h1 < h2)),
        ((a1 and a2 == -1) and (h1 > h2)),
        ((a1 < a2) and (h1 and h2 == -1)),
        ((a1 > a2) and (h1 and h2 == -1)),
        # directionals
        (a1 == a2 and h1 == h2),
        (a1 < a2 and h1 == h2),
        (a1 > a2 and h1 == h2),
        (a1 == a2 and h1 < h2),
        (a1 == a2 and h1 > h2),
        (a1 < a2 and h1 < h2),
        (a1 > a2 and h1 > h2),
        (a1 < a2 and h1 > h2),
        (a1 > a2 and h1 < h2)
    ]
    return propositions


def main():
    # data = lines(["nba"], output='dict')
    # print(data)
    # print(len(data))
    # return data
    row = single_game_line()
    print(row)
    return row


if __name__ == '__main__':
    data = main()
