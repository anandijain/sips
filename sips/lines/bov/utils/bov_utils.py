'''
utils functions for bov.py
'''

import time
import json

import requests as r
from requests_futures.sessions import FuturesSession

from pydash import at

import sips.h.macros as m
from sips.lines.bov import bov_main

BOV_URL = 'https://www.bovada.lv/services/sports/event/v2/events/A/description/'
BOV_SCORES_URL = 'https://services.bovada.lv/services/sports/results/api/v1/scores/'

MKT_TYPE = {
    'Point Spread': 'ps',
    'Runline': 'ps',
    'Moneyline': 'ml',
    'Total': 'tot'
}

TO_GRAB = {
    'ps': ['american', 'handicap'],  # spread
    'ml': ['american'],
    'tot': ['american', 'handicap'],
    'competitors': ['home', 'id', 'name']
}

PRICE_LABELS = {
    'ps': ['a_ps', 'h_ps', 'a_hcap', 'h_hcap'],
    'ml': ['a_ml', 'h_ml'],
    'tot': ['a_tot', 'h_tot', 'a_hcap_tot', 'h_hcap_tot', 'a_ou', 'h_ou']
}


def reduce_mkt_type(market_desc):
    try:
        reduced = MKT_TYPE[market_desc]
    except KeyError:
        # print(f'{market_desc} not supported')
        return 'ml'
    return reduced


def parse_json(json, keys, output='dict'):
    '''
    input: dictionary and list of strings
    returns dict

    if the key does not exist in the json
    the key is still created with None as the value
    '''
    data = {}
    json_keys = json.keys()
    for j_key in json_keys:
        if j_key in keys:
            d = json.get(j_key)
            data[j_key] = d
    if output == 'list':
        return list(data.values())
    elif output == 'dict':
        return data
    else:
        return None


def parse_display_groups(event):
    '''

    '''
    groups = event['displayGroups']
    full_set = {}
    for group in groups:
        desc = group.get('description')
        if not desc:
            continue
        cleaned = clean_desc(desc)
        data_dict = parse_display_group(group)
        full_set[cleaned] = data_dict

    print(f'full_set: {full_set}')
    return full_set


def parse_display_group(display_group):
    group_markets = display_group.get('markets')
    data = parse_markets(group_markets)
    return data


def parse_event(event, verbose=False):
    '''
    [sport, game_id, a_team, h_team, last_mod, num_markets, live],
    [quarter, secs, a_pts, h_pts, status], [
    a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot,
    a_hcap_tot, h_hcap_tot, a_ou, h_ou, game_start_time]
    '''
    sport, game_id, last_mod, num_markets, live = parse_json(event, [
        'sport', 'id', 'lastModified', 'numMarkets', 'live'], output='list')
    a_team, h_team = teams(event)
    quarter, secs, a_pts, h_pts, status = score(game_id)

    display_groups = event['displayGroups'][0]
    markets = display_groups['markets']
    a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, \
        h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou = parse_markets(markets)

    game_start_time = event['startTime']
    # todo reformat schema [sport, game_id, a_team, h_team, a_ml, h_ml, a_pts, h_pts, quarter, secs, ...]
    ret = [sport, game_id, a_team, h_team, last_mod, num_markets, live,
           quarter, secs, a_pts, h_pts, status,
           a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot,
           a_hcap_tot, h_hcap_tot, a_ou, h_ou, game_start_time]

    if verbose:
        print(f'event: {ret}')
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
    all_markets = {}

    # a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot, \
    #     a_hcap_tot, h_hcap_tot, a_ou, h_ou = ["NaN" for _ in range(12)]

    for market in markets:
        mkt_type = reduce_mkt_type(market.get('description'))
        print(f'mkt_type: {mkt_type}')
        if not mkt_type:
            continue

        period = market.get('period')
        period_info = at(period, 'description', 'abbreviation', 'live')

        cleaned = clean_desc(period_info[0])
        mkt_key = mkt_type + '_' + cleaned

        market_data = parse_market(market)
        all_markets[mkt_key] = market_data

    return all_markets


def parse_market(market):
    '''
    given: market in bovada sport json
    returns: dictionary w (field , field_value)
    '''
    period = market.get('period')
    is_live = int(period['live'])
    print(f'is_live: {is_live}')
    mkt_type = reduce_mkt_type(market.get('description'))

    outcomes = market.get('outcomes')

    if mkt_type == 'ps':
        data = spread(outcomes)
        print(f'{mkt_type} data: {data}')
    elif mkt_type == 'ml':
        data = ml_no_teams(outcomes)
        print(f'{mkt_type} data: {data}')
    elif mkt_type == 'tot':
        data = total(outcomes)
        print(f'{mkt_type} data: {data}')
    else:
        print(f'in parse markets mkt_type: {mkt_type}')
    ret = data
    if isinstance(data, dict):
        ret = []
        for v in data.values():
            ret += v
        print(f'this RET: {ret}')
    
    ret.append(is_live)
    print(f'ret: {ret}')
    return ret


def clean_desc(desc):
    to_replace = [('-', ''), ('  ', ' '), (' ', '_')]
    ret = desc.lower()
    for tup in to_replace:
        ret = ret.replace(tup[0], tup[1])
    return ret


def spread(outcomes):
    '''
    gets both teams spread data
    '''
    a_ps, a_hcap, h_ps, h_hcap = ['NaN' for _ in range(4)]
    for outcome in outcomes:
        price = outcome['price']
        if outcome['type'] == 'A':
            a_ps, a_hcap = parse_json(price, TO_GRAB['ps'], 'list')
        else:
            h_ps, h_hcap = parse_json(price, TO_GRAB['ps'], 'list')

    print(f'a_ps, h_ps, a_hcap, h_hcap: {a_ps, h_ps, a_hcap, h_hcap}')
    return [a_ps, h_ps, a_hcap, h_hcap]


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
    print(f'a_ml, h_ml: {a_ml, h_ml}')
    return [a_ml, h_ml]


def ml_no_teams(outcomes):
    mls = {}
    for oc in outcomes:
        competitor_id = oc.get('competitorId')
        desc = oc.get('description')
        cleaned = clean_desc(desc)
        price = oc.get('price')
        american = price['american']
        mls[competitor_id] = [cleaned, american]
    return mls


def total_no_teams():    
    pass


def total(outcomes):
    '''
    gets the over_under
    '''
    if not outcomes:
        return ['NaN' for _ in range(6)]

    a_outcome = outcomes[0]
    a_ou = a_outcome.get('type')
    a_price = a_outcome.get('price')
    a_tot, a_hcap_tot = parse_json(a_price, TO_GRAB['tot'], 'list')

    h_outcome = outcomes[1]
    h_ou = h_outcome.get('type')
    h_price = h_outcome.get('price')
    h_tot, h_hcap_tot = parse_json(h_price, TO_GRAB['tot'], 'list')

    print(
        f'[a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou]: {[a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou]}')

    return [a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou]


def competitors(competitors, verbose=True):
    '''
    keys: 'home' (bool), 'id' (str), 'name' (str)
    '''
    data = [parse_json(t, TO_GRAB['competitors']) for t in competitors]
    if verbose:
        print(f'competitors: {data}') 
    return data  # list of two dictionaries


def teams(event):
    '''
    returns away, home team names (str)
    '''
    comps = event.get('competitors')
    if not comps:
        return 'NaN', 'NaN'
    teams = competitors(comps)
    a_team, h_team = [team['name'] for team in teams]
    return a_team, h_team


def bov_comp_ids(event):
    '''
    get competitor ids
    '''
    comps = event.get('competitors')
    if not comps:
        return 'NaN', 'NaN'
    teams = competitors(comps)
    a_id, h_id = [team['id'] for team in teams]
    return a_id, h_id


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


def req_json(sport='football/nfl', url=None, sleep=0.25):
    '''
    requests the bovada link, and specific sport as arg
    '''
    if not url:
        url = BOV_URL + sport
        print(f'no url given. url was set to: {url}')

    bov_json = r.get(url).json()
    time.sleep(sleep)
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


def dict_from_events(events, key='id', rows=True):
    '''
    returns a dictionary of (key, event) or (key, list)

    key must be in the event json data
    rows: (bool)
        - if true, set key-vals to rows
    '''
    print(f'len events: {len(events)}')
    event_dict = {e[key]: parse_event(e) if rows else e for e in events}

    return event_dict


def list_from_jsons(jsons, rows=False):
    '''
    jsons is a list of dictionaries for each sport 
    returns a list of events
    '''
    events = [parse_event(
        e) if rows else e for j in jsons for e in json_events(j)]
    return events


def json_events(json_dict):
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

if __name__ == "__main__":
    bov_main.main()