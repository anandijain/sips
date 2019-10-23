'''
uses the bovada api to get json data for odds and scores
'''
import time
import json

import requests as r
from requests_futures.sessions import FuturesSession

import sips.h.macros as m


BOV_URL = 'https://www.bovada.lv/services/sports/event/v2/events/A/description/'
BOV_SCORES_URL = 'https://services.bovada.lv/services/sports/results/api/v1/scores/'


def async_req(links):
    '''
    asyncronous request of the list of links
    '''
    session = FuturesSession()
    jsons = [session.get(link).result().json() for link in links]
    return jsons


def get_ids(events):
    '''
    returns the ids for all bov events given
    '''
    if not events:
        events = get_events()
    ids = []
    for event in events:
        game_id = event['id']
        ids.append(game_id)
    return ids


def games(config_path):
    '''
    reading in config, we return the lines for the games specified
    '''
    with open(config_path) as config:
        conf = json.load(config)
    ids = conf['games'][0]['game_ids']
    bov_events = get_events()
    ret = []
    for game_id in ids:
        for event in bov_events:
            if int(event['id']) == game_id:
                ret.append(parse_event(event))
    return ret


def req_json(link=BOV_URL, sport='football/nfl'):
    '''
    requests the bovada link, and specific sport as arg
    '''
    full_link = link + sport
    print(full_link)
    bov_json = r.get(full_link).json()
    return bov_json


def events_sports():
    '''
    gets all events for all the sports specified in macros.py
    '''
    json_data = async_req(m.build_urls())
    bov_events = []
    for data in json_data:
        if not data:
            continue
        events = data[0]['events']
        bov_events += events
    return bov_events


def get_events(sport='mlb'):
    '''
    given a specific sport, return the events
    '''
    try:
        sport = m.league_to_sport_and_league[sport]
    except KeyError:
        print('forcing nfl')
        sport = m.league_to_sport_and_league['nfl']

    json_data = req_json(sport=sport)
    print(json_data)
    bov_events = json_data[0]['events']
    return bov_events


def lines(events=None, sport='mlb'):
    '''
    returns the lines for a given sport
    '''
    if not events:
        events = get_events(sport=sport)
    rows = [parse_event(e) for e in events]
    return rows


def header():
    '''
    column names for dataframe
    '''
    return ['sport', 'game_id', 'a_team', 'h_team', 'cur_time', 'last_mod', 'num_markets', 'live', \
        'quarter', 'secs', 'a_pts', 'h_pts', 'status', \
        'a_ps', 'h_ps', 'a_hcap', 'h_hcap', 'a_ml', 'h_ml', 'a_tot', 'h_tot', \
        'a_hcap_tot', 'h_hcap_tot', 'a_ou', 'h_ou', 'game_start_time']


def parse_event(event):
    '''
    [sport, game_id, a_team, h_team, cur_time, last_mod, num_markets, live],
    [quarter, secs, a_pts, h_pts, status], [
    a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot,
    a_hcap_tot, h_hcap_tot, a_ou, h_ou, game_start_time]
    '''
    sport = event['sport']
    game_id = event['id']
    a_team, h_team = teams(event)
    cur_time = time.time()
    last_mod = event['lastModified']
    num_markets = event['numMarkets']
    live = event['live']
    quarter, secs, a_pts, h_pts, status = score(game_id)


    display_groups = event['displayGroups'][0]
    markets = display_groups['markets']
    a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, \
    h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou = parse_markets(markets)

    game_start_time = event['startTime']

    ret = [sport, game_id, a_team, h_team, cur_time, last_mod, num_markets, live, \
        quarter, secs, a_pts, h_pts, status, \
        a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot, \
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

    data = [a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot, \
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


def game_json(game_id):
    '''
    requests the json data for the specific game for score data
    '''
    json_data = r.get(BOV_SCORES_URL + game_id).json()
    time.sleep(0.05)
    return json_data


def score(game_id):
    '''
    given a game_id, returns the score data of the game
    '''
    [quarter, secs, a_pts, h_pts, status] = ['NaN' for _ in range(5)]

    json_data = game_json(game_id)
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

if __name__ == '__main__':
    RET = lines()
    print(RET)
