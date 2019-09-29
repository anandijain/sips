import requests as r

import time

import json

bov ="https://www.bovada.lv/services/sports/event/v2/events/A/description/football/nfl?marketFilterId=def&eventsLimit=50&lang=en"
bov_scores_url = "https://services.bovada.lv/services/sports/results/api/v1/scores/"

def get_bov_ids(events):
    if not events:
        events = get_bov_events()
    ids = []
    for event in events:
        game_id = event['id']
        ids.append(game_id)
    return ids

def get_bov_games(config_path):
    with open(config_path) as config:
        conf = json.load(config)
    ids = conf['games'][0]['game_ids']
    bov_events = get_bov_events()
    ret = []
    for game_id in ids:
        for event in bov_events:
            if int(event['id']) == game_id:
                ret.append(bov_line(event))
    return ret

def bov_json():
    bov_json = r.get(bov).json()
    return bov_json

def get_bov_events():
    json = bov_json()
    bov_events = json[0]['events']
    return bov_events

def bov_lines(events=None):
    if not events:
        events = get_bov_events()
    lines = []
    for event in events:
        lines.append(bov_line(event))
    return lines

def bov_line(event):
    '''
    [sport, game_id, a_team, h_team, cur_time, last_mod, num_markets, live],
    [quarter, secs, a_pts, h_pts, status], [
    a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot,
    a_hcap_tot, h_hcap_tot, a_ou, h_ou, game_start_time]
    '''
    sport = event['sport']
    game_id = event['id']
    a_team, h_team = bov_teams(event)
    cur_time = time.time()
    last_mod = event['lastModified']
    num_markets = event['numMarkets']
    live = event['live']
    quarter, secs, a_pts, h_pts, status = bov_score(game_id)


    display_groups = event['displayGroups'][0]
    markets = display_groups['markets']
    a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou = bov_markets(markets)

    game_start_time = event['startTime']

    ret = [sport, game_id, a_team, h_team, cur_time, last_mod, num_markets, live, \
        quarter, secs, a_pts, h_pts, status, \
        a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot, \
        a_hcap_tot, h_hcap_tot, a_ou, h_ou, game_start_time]

    return ret

def bov_markets(markets):
    for market in markets:
        desc = market['description']
        outcomes = market['outcomes']

        if desc == 'Point Spread':
            a_ps, h_ps, a_hcap, h_hcap = parse_bov_spread(outcomes)
        elif desc == 'Moneyline':
            a_ml, h_ml = parse_bov_ml(outcomes)
        elif desc == 'Total':
            a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou = parse_bov_tot(outcomes)

    return (a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot,
            a_hcap_tot, h_hcap_tot, a_ou, h_ou)


def parse_bov_spread(outcomes):
    for outcome in outcomes:
        price = outcome['price']
        if outcome['type'] == 'A':
            a_ps = price['american']
            a_hcap = price['handicap']
        else:
            h_ps = price['american']
            h_hcap = price['handicap']
    return a_ps, h_ps, a_hcap, h_hcap

def parse_bov_ml(outcomes):
    for outcome in outcomes:
        price = outcome['price']
        if outcome['type'] == 'A':
            a_ml = price['american']
        else:
            h_ml = price['american']
    return a_ml, h_ml

def parse_bov_tot(outcomes):
    a_price = outcomes[0]['price']
    h_price = outcomes[1]['price']
    a_tot = a_price['american']
    h_tot = h_price['american']
    a_hcap_tot = a_price['handicap']
    h_hcap_tot = h_price['handicap']
    a_ou = outcomes[0]['type']
    h_ou = outcomes[1]['type']
    return a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou


def bov_teams(event):
    # returns away, home
    team_one = event['competitors'][0]
    team_two = event['competitors'][1]
    if team_one['home']:
        h_team = team_one['name']
        a_team = team_two['name']
    else:
        a_team = team_one['name']
        h_team = team_two['name']
    return a_team, h_team

def bov_game_json(game_id):
    game_json = r.get(bov_scores_url + game_id).json()
    time.sleep(0.05)
    return game_json

def bov_score(game_id):
    json = bov_game_json(game_id)
    clock = json['clock']
    quarter = clock['periodNumber']
    secs = clock['relativeGameTimeInSecs']
    a_pts = json['latestScore']['visitor']
    h_pts = json['latestScore']['home']
    status = 0
    if json['gameStatus'] == "IN_PROGRESS":
        status = 1
    return (quarter, secs, a_pts, h_pts, status)
