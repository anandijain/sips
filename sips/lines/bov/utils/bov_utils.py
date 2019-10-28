'''
utils functions for bov.py
'''

import time
import requests as r

# from pydash import at

import sips.h.macros as m
import sips.h.openers as o

from sips.lines.bov import bov_main

HEADERS = {'User-Agent': 'Mozilla/5.0'}


MKT_TYPE = {
    'Point Spread': 'ps',
    'Runline': 'ps',
    'Moneyline': 'ml',
    'Total': 'tot'
}

TO_GRAB = {
    'ps': ['american', 'handicap'],
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


def merge_lines_scores(lines, scores):
    '''
    both type dict
    '''
    ret = {}
    for k, v in lines.items():
        score_data = scores.get(k)
        if not score_data:
            score_data = [None for _ in range(5)]
        row = v[0] + score_data + v[1]
        ret[k] = row

    return ret


def dict_from_events(events, key='id', rows=True, grab_score=False):
    '''
    returns a dictionary of (key, event) or (key, list)

    key must be in the event json data
    rows: (bool)
        - if true, set key-vals to rows
    '''
    event_dict = {e[key]: parse_event(
        e, grab_score=grab_score) if rows else e for e in events}
    return event_dict


def parse_event(event, verbose=False, grab_score=True):
    '''
    [sport, game_id, a_team, h_team, last_mod, num_markets, live],
    [quarter, secs, a_pts, h_pts, status], [
    a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot,
    a_hcap_tot, h_hcap_tot, a_ou, h_ou, game_start_time]
    '''
    game_id, sport, live, num_markets, last_mod = parse_json(
        event, ['id', 'sport', 'live', 'numMarkets', 'lastModified'],
        output='list')

    a_team, h_team = teams(event)

    display_groups = event['displayGroups'][0]
    markets = display_groups['markets']
    a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, \
        h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou = grab_row_from_markets(
            markets)

    game_start_time = event['startTime']

    if not grab_score:
        section_1 = [sport, game_id, a_team,
                     h_team, last_mod, num_markets, live]
        section_2 = [a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot,
                     a_hcap_tot, h_hcap_tot, a_ou, h_ou, game_start_time]
        ret = [section_1, section_2]
    else:
        score_url = m.BOV_SCORES_URL + game_id
        score_data = req_json(score_url)

        quarter, secs, a_pts, h_pts, status = score(score_data)
        ret = [sport, game_id, a_team, h_team, last_mod, num_markets, live,
               quarter, secs, a_pts, h_pts, status, a_ps, h_ps, a_hcap,
               h_hcap, a_ml, h_ml, a_tot, h_tot, a_hcap_tot, h_hcap_tot,
               a_ou, h_ou, game_start_time]

    # ret = [sport, game_id, a_team, h_team, a_pts, h_pts, a_ml, h_ml,
    #         quarter, secs, status, num_markets, live, a_ps, h_ps, a_hcap,
    #         h_hcap, a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou,
    #         game_start_time, last_mod]

    if verbose:
        print(f'event: {section_1} {section_2}')

    return ret


def teams_from_line(line):
    '''
    return the a_team and h_team indices in row list
    '''
    return line[2:4]


def grab_row_from_markets(markets):
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


def parse_markets(markets, output='list'):
    '''
    parse market dict in bov event
    '''
    all_markets = {}

    # a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot, \
    #     a_hcap_tot, h_hcap_tot, a_ou, h_ou = ["NaN" for _ in range(12)]

    for market in markets:
        mkt_type = reduce_mkt_type(market.get('description'))
        if not mkt_type:
            continue

        period = market.get('period')
        to_grab = ['description', 'abbreviation', 'live']
        period_info = parse_json(period, to_grab, 'list')

        cleaned = clean_desc(period_info[0])
        mkt_key = mkt_type + '_' + cleaned

        market_data = parse_market(market)
        all_markets[mkt_key] = market_data

    if output == 'list':
        return list(all_markets.values())
    return all_markets


def parse_market(market):
    '''
    given: market in bovada sport json
    returns: dictionary w (field , field_value)
    '''
    period = market.get('period')
    is_live = int(period['live'])
    mkt_type = reduce_mkt_type(market.get('description'))

    outcomes = market.get('outcomes')

    if mkt_type == 'ps':
        data = spread(outcomes)
    elif mkt_type == 'tot':
        data = total(outcomes)
    else:
        data = ml_no_teams(outcomes)

    ret = data

    if isinstance(data, dict):
        ret = []
        for v in data.values():
            ret += v
        print(f'this RET: {ret}')

    ret.append(is_live)
    return ret


def clean_desc(desc):
    '''

    '''
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
    # print(f'a_ml, h_ml: {a_ml, h_ml}')
    return [a_ml, h_ml]


def ml_no_teams(outcomes):
    '''

    '''
    mls = {}
    for oc in outcomes:
        competitor_id = oc.get('competitorId')
        desc = oc.get('description')
        cleaned = clean_desc(desc)
        price = oc.get('price')
        american = price['american']
        mls[competitor_id] = [cleaned, american]
    return mls


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

    return [a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou]


def competitors(competitors, verbose=False):
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


def get_scores(events, session=None):
    '''
    quarter, secs, a_pts, h_pts, status
    '''
    ids = get_ids(events)
    # links = {game_id : m.BOV_SCORES_URL + game_id for game_id in ids}

    links = [m.BOV_SCORES_URL + game_id for game_id in ids]
    raw = o.async_req_dict(links, 'eventId', session=session)
    scores_dict = {g_id: score(j) for g_id, j in raw.items()}
    return scores_dict


def score(json_data):
    '''
    given a game_id, returns the score data of the game
    '''
    [quarter, secs, a_pts, h_pts, game_status] = ['NaN' for _ in range(5)]

    if not json_data:
        return [quarter, secs, a_pts, h_pts, game_status]

    clock = json_data.get('clock')
    if clock:
        quarter = clock['periodNumber']
        secs = clock['relativeGameTimeInSecs']
    latest_score = json_data.get('latestScore')

    if not latest_score:
        a_pts = 0
        h_pts = 0
    else:
        a_pts = latest_score.get('visitor')
        h_pts = latest_score.get('home')

    status = json_data.get('gameStatus')
    if status:
        game_status = status
    else:
        status = None
    return [quarter, secs, a_pts, h_pts, game_status]


def req_json(url, sleep=0.5, verbose=False):
    '''
    requests the link, returns json
    '''
    try:
        req = r.get(url)
    except:
        return None

    time.sleep(sleep)

    try:
        bov_json = req.json()
    except:
        print(f'{url} had no json')
        return None

    if verbose:
        print(f"req'd url: {url}")
    return bov_json


def get_ids(events):
    '''
    returns the ids for all bov events given
    '''
    ids = []
    for event in events:
        game_id = event.get('id')
        if game_id:
            ids.append(game_id)
    return ids


def list_from_jsons(jsons, rows=False):
    '''
    jsons is a list of dictionaries for each sport 
    returns a list of events
    '''
    events = [parse_event(e)
              if rows else e for j in jsons for e in json_events(j)]
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
        sport = m.SPORT_TO_SUFFIX[sport]
    except KeyError:
        print('forcing nfl')
        sport = m.SPORT_TO_SUFFIX['nfl']
    return sport


def header():
    '''
    column names for dataframe
    '''
    return ['sport', 'game_id', 'a_team', 'h_team', 'last_mod', 'num_markets',
            'live', 'quarter', 'secs', 'a_pts', 'h_pts', 'status', 'a_ps',
            'h_ps', 'a_hcap', 'h_hcap', 'a_ml', 'h_ml', 'a_tot', 'h_tot',
            'a_hcap_tot', 'h_hcap_tot', 'a_ou', 'h_ou', 'game_start_time']


def bov_all_dict():
    '''

    '''
    all_dict = {}
    req = r.get('https: // www.bovada.lv/services/sports /\'
                'event/v2/events/A/description/basketball/nba').json()
    es = req[0].get('events')
    for event in es:
        desc = event.get('description')
        # print(f'desc: {desc}')
        if not desc:
            continue
        event_dict = utils.parse_display_groups(event)
        cleaned = utils.clean_desc(desc)
        all_dict[cleaned] = event_dict
    # print(f'all_dict: {all_dict}')
    return all_dict
    

if __name__ == "__main__":
    bov_main.main()
