'''
utils functions for bov.py
'''

import time
import json

import requests as r
from requests_futures.sessions import FuturesSession

import sips.h.macros as m

BOV_URL = 'https://www.bovada.lv/services/sports/event/v2/events/A/description/'
BOV_SCORES_URL = 'https://services.bovada.lv/services/sports/results/api/v1/scores/'


def parse_event(event):
    '''
    [sport, game_id, a_team, h_team, last_mod, num_markets, live],
    [quarter, secs, a_pts, h_pts, status], [
    a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot,
    a_hcap_tot, h_hcap_tot, a_ou, h_ou, game_start_time]
    '''
    sport = event['sport']
    game_id = event['id']
    a_team, h_team = teams(event)
    last_mod = event['lastModified']
    num_markets = event['numMarkets']
    live = event['live']
    quarter, secs, a_pts, h_pts, status = score(game_id)

    display_groups = event['displayGroups'][0]
    markets = display_groups['markets']
    a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, \
        h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou = parse_markets(markets)

    game_start_time = event['startTime']

    ret = [sport, game_id, a_team, h_team, last_mod, num_markets, live,
           quarter, secs, a_pts, h_pts, status,
           a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot,
           a_hcap_tot, h_hcap_tot, a_ou, h_ou, game_start_time]

    return ret


def teams_from_line(line):
    '''
    return the a_team and h_team indices in row list
    '''
    return line[2:4]


def parse_markets(markets):
    '''
    parse market dict in bov event
    '''
    a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot, \
        a_hcap_tot, h_hcap_tot, a_ou, h_ou = ["NaN" for _ in range(12)]
    for market in markets:
        desc = market['description']
        outcomes = market['outcomes']
        if desc == 'Point Spread':
            a_ps, h_ps, a_hcap, h_hcap = spread(outcomes)
        elif desc == 'Moneyline':
            a_ml, h_ml = moneyline(outcomes)
        elif desc == 'Total':
            a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou = total(outcomes)

    data = [a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot,
            a_hcap_tot, h_hcap_tot, a_ou, h_ou]
    return data


def spread(outcomes):
    '''
    gets both teams spread data
    '''
    a_ps, a_hcap, h_ps, h_hcap = ['NaN' for _ in range(4)]
    for outcome in outcomes:
        price = outcome['price']
        if outcome['type'] == 'A':
            a_ps = price['american']
            a_hcap = price['handicap']
        else:
            h_ps = price['american']
            h_hcap = price['handicap']
    return a_ps, h_ps, a_hcap, h_hcap


def moneyline(outcomes):
    '''
    gets both teams moneyline
    '''
    a_ml = 'NaN'
    h_ml = 'NaN'
    for outcome in outcomes:
        price = outcome['price']
        if outcome['type'] == 'A':
            a_ml = price['american']
        else:
            h_ml = price['american']

    return a_ml, h_ml


def total(outcomes):
    '''
    gets the over_under
    '''
    if not outcomes:
        return ['NaN' for _ in range(6)]
    a_price = outcomes[0]['price']
    h_price = outcomes[1]['price']
    a_tot = a_price['american']
    h_tot = h_price['american']
    a_hcap_tot = a_price['handicap']
    h_hcap_tot = h_price['handicap']
    a_ou = outcomes[0]['type']
    h_ou = outcomes[1]['type']
    return [a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou]


def teams(event):
    '''
    returns away, home team names (str)
    '''
    team_one = event['competitors'][0]
    team_two = event['competitors'][1]
    if team_one['home']:
        h_team = team_one['name']
        a_team = team_two['name']
    else:
        a_team = team_one['name']
        h_team = team_two['name']
    return a_team, h_team


def score(game_id):
    '''
    given a game_id, returns the score data of the game
    '''
    [quarter, secs, a_pts, h_pts, status] = ['NaN' for _ in range(5)]
    
    game_url = BOV_SCORES_URL + game_id
    json_data = req_json(url=game_url)

    if json_data.get('Error'):
        return [quarter, secs, a_pts, h_pts, status]
    clock = json_data.get('clock')
    if clock:
        quarter = clock['periodNumber']
        secs = clock['relativeGameTimeInSecs']

    a_pts = json_data['latestScore']['visitor']
    h_pts = json_data['latestScore']['home']
    status = 0
    if json_data['gameStatus'] == "IN_PROGRESS":
        status = 1

    return [quarter, secs, a_pts, h_pts, status]


def async_req(links):
    '''
    asyncronous request of list of links
    '''
    session = FuturesSession()
    jsons = [session.get(link).result().json() for link in links]
    return jsons


def req_json(sport='football/nfl', url=None):
    '''
    requests the bovada link, and specific sport as arg
    '''
    if not url:
        full_link = BOV_URL + sport
        print(f'no url given. url was set to: {full_link}')

    bov_json = r.get(full_link).json()
    return bov_json


def get_ids(events):
    '''
    returns the ids for all bov events given
    '''
    ids = []
    for event in events:
        game_id = event['id']
        ids.append(game_id)
    return ids


def event_dict_from_jsons(jsons, key='id'):
    '''
    constructs a dictionary of events
    key must be in the event json data
    '''
    events = events_from_jsons(jsons)
    i = 0
    print(f'events{[i]}: {events[i]}')
    event_dict = {event[key]: event for event in events}

    for event in events:
        game_id = event[key]
        event_dict[game_id] = event

    return event_dict


def events_from_jsons(jsons):
    '''
    jsons is a list of dictionaries for each sport 
    returns a list of events
    '''
    bov_events = [event for j in jsons for event in events_from_json(j)]
    return bov_events


def events_from_json(json_dict):
    '''
    simply accesses the events in the json
    '''
    if not json_dict:
        return []
    events = json_dict[0]['events']
    return events


def match_sport_str(sport='mlb'):
    '''
    maps string to the url suffix to bovada api
    '''
    try:
        sport = m.sport_to_suffix[sport]
    except KeyError:
        print('forcing nfl')
        sport = m.sport_to_suffix['nfl']
    return sport


def header():
    '''
    column names for dataframe
    '''
    return ['sport', 'game_id', 'a_team', 'h_team', 'last_mod', 'num_markets', 'live',
            'quarter', 'secs', 'a_pts', 'h_pts', 'status', 'a_ps', 'h_ps', 'a_hcap',
            'h_hcap', 'a_ml', 'h_ml', 'a_tot', 'h_tot', 'a_hcap_tot', 'h_hcap_tot', 'a_ou',
            'h_ou', 'game_start_time']
