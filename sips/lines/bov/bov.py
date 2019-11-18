'''
uses the bovada api to get json data for odds and scores
'''
import requests as r

import pandas as pd

import numpy as np

import sips.h.grab as g

from sips.macros import bov as bm
from sips.lines.bov.utils import bov_utils as u
from sips.macros import nfl
from sips.macros import nba
from sips.macros import nhl
import sips.h.helpers as h


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
    row = list(row)
    teams = row[2:4]

    for t in teams:
        hot_teams = teams_dict[t]
        ret += hot_teams

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


def prep_game_dataset(fn, sports=['nba']): # , zip_data=True, verbose=False):
    teams_dict, statuses_dict = dicts_for_one_hotting(sports)

    df = pd.read_csv(fn)
    prev = [None, None]
    prev_row = [None for _ in range(25)]
    X = []
    y = []
    for i, row in df.iterrows():
            
        cur_row = row.values
        cur_ml = list(row[['a_ml', 'h_ml']])
        if i == 0:
            prev_ml = cur_ml
            prev_row = cur_row
            continue

        x = serialize_row(prev_row, teams_dict, statuses_dict)
        transition_class = classify_transition(prev_ml, cur_ml)
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
    X = np.reshape(np.concatenate(X, axis=0), (len_game, 1, -1))
    # y = np.reshape(np.concatenate(y, axis=0), (466, 1, -1))

    return X, y


def game_transitions(game, verbose=False):
    '''
    given a dataframe of live lines for a single game,
    returns a list of classifications for the line movement
    '''
    transition_classes = []
    teams_dict, statuses_dict = dicts_for_one_hotting()

    prev = [None, None]

    for i, row in game.iterrows():
        cur = list(row[['a_ml', 'h_ml']])
        transition_class = classify_transition(prev, cur)
        transition_classes.append(transition_class)
        prev = cur
    
    if verbose:
        strings = {i : s for i, s in enumerate(bm.TRANSITION_CLASS_STRINGS)}

        for i, t in enumerate(transition_classes):
            class_num = np.argmax(t)
            print(f'{i}: {strings[class_num]}')

    return transition_classes


def dicts_for_one_hotting(sports=['nfl', 'nba', 'nhl']):
    team_list = [] 

    for s in sports:
        if s == 'nfl':
            team_list += nfl.teams
        elif s == 'nba':
            team_list += nba.teams
        elif s == 'nhl':
            team_list += nhl.teams

    teams_dict = h.hot_list(team_list, output='list')
    statuses = ['GAME_END', 'HALF_TIME', 'INTERRUPTED',
                'IN_PROGRESS', 'None', 'PRE_GAME']
    statuses_dict = h.hot_list(statuses, output='list')
    return teams_dict, statuses_dict


def parse_row_mls(ml_list):
    '''
    given a list of unparsed moneylines (eg can be 'EVEN' and None)
    edit the values such that 'EVEN' -> 100 and None -> -1
    typical order of list is [a0, h0, a1, h1]
    '''
    ret = []
    for line in ml_list:
        if line == 'EVEN':
            ret.append(100)
        elif line == None:
            ret.append(-1)
        else:
            try:
                x = float(line)
            except:
                x = -1
            ret.append(x)
    return ret


def classify_transition(prev_mls, cur_mls):
    '''
    uses the propositions described in transitions() to return a numpy array
    with the class of transition corresponding to the input moneylines
    '''

    mls = parse_row_mls(prev_mls + cur_mls)
    a_prev = mls[0]
    h_prev = mls[1]
    a_cur = mls[2]
    h_cur = mls[3]

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
