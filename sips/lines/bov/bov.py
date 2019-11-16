'''
uses the bovada api to get json data for odds and scores
'''
import requests as r
import sips.h.grab as g
import sips.h.helpers as h
from sips.macros import nfl
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


def serialize_row(row):
    '''
    going to take in something like this:
    ['FOOT', 5741304, 'Pittsburgh Steelers', 'Cleveland Browns', 1573540736617, 28, 
    False, '0', '-1', '0', '0', 'PRE_GAME', '2.5', '-2.5', '-105', '-115', '+125', 
    '-145', '40.0', '40.0', '-110', '-110', 'O', 'U', 1573780800000]
    and return a np array 
    '''
    nfl_teams = nfl.teams
    hotted_team_dict = h.hot_list(nfl_teams)
    teams = row[2:4]
    hot_teams = [hotted_team_dict[t] for t in teams]
    return hot_teams

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
