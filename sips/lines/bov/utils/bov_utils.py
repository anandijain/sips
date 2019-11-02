'''
utils functions for bov.py
'''

import time
import requests as r

import sips.h.fileio as io
import sips.h.grab as g
from sips.macros import bov as bm
from sips.macros import macros as m
from sips.lines import lines as ll
from sips.lines.bov.utils import bov_utils as u


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

    # print(f'full_set: {full_set}')
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


def events_from_jsons(jsons):
    '''
    jsons is a list of dictionaries for each sport 
    if rows, return parsed row data instead of list of events
    '''
    events = [e for j in jsons for e in events_from_json(j)]
    return events


def rows_from_jsons(jsons):
    '''
    jsons is a list of dictionaries for each sport 
    if rows, return parsed row data instead of list of events
    '''
    events = [u.parse_event(e) for j in jsons
              for e in events_from_json(j)]
    return events


def dict_from_events(events, key='id', rows=True):
    '''
    returns a dictionary of (key, event) or (key, list)

    key must be in the event json data
    rows: (bool)
        - if true, set key-vals to rows
    '''
    event_dict = {e[key]: parse_event(e) if rows else e for e in events}
    return event_dict


def parse_event(event, verbose=False):
    '''
    parses an event with three markets (spread, ml, total)
    returns list of data following the order in header()
    '''
    game_id, sport, live, num_markets, last_mod = parse_json(
        event, ['id', 'sport', 'live', 'numMarkets', 'lastModified'],
        output='list')
    a_team, h_team = teams(event)
    game_start_time = event['startTime']

    events_all_mkts = parse_display_groups(event)
    game_lines = events_all_mkts.get('game_lines')

    if not game_lines:
        print(f'events_all_mkts: {events_all_mkts}')
        a_ps, h_ps, a_hcap, h_hcap, ps_M_live, a_team, a_ml, h_team, \
            h_ml, ml_M_live, a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, \
            h_ou, tot_M_live = [None for _ in range(17)]
    else:
        ml_M = game_lines['moneyline_ml_M']
        ps_M = game_lines['point_spread_ps_M']
        tot_M = game_lines['total_tot_M']

        a_ps, h_ps, a_hcap, h_hcap, ps_M_live = ps_M
        a_team, a_ml, h_team, h_ml, ml_M_live = ml_M
        a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou, tot_M_live = tot_M

    score_url = bm.BOV_SCORES_URL + game_id
    score_data = g.req_json(score_url)
    quarter, secs, a_pts, h_pts, status = score(score_data)

    ret = [sport, game_id, a_team, h_team, last_mod, num_markets, live,
            quarter, secs, a_pts, h_pts, status, a_ps, h_ps, a_hcap,
            h_hcap, a_ml, h_ml, a_tot, h_tot, a_hcap_tot, h_hcap_tot,
            a_ou, h_ou, game_start_time]

    return ret


def grab_row_from_markets(markets):
    '''
    to be deprecated, only grabs the filtered mkts
    parse main markets (match ps, ml, totals) json in bov event
    '''
    a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot, \
        a_hcap_tot, h_hcap_tot, a_ou, h_ou = ["NaN" for _ in range(12)]
    for market in markets:
        desc = market.get('description')
        period_desc, abbrv, live = mkt_period_info(market)
        if period_desc == 'Match':        
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


def parse_markets(markets, output='dict'):
    '''
    parse markets in bov event
    keys of all_markets are: cleaned_mkt_desc + reduced mkt type + period abbrv

    key examples
    Moneyline -> moneyline_ml_M
    Point Spread -> point_spread_ps_M
    First Team to reach 20 points -> first_team_to_reach_20_points_ml_M
    '''
    all_markets = {}

    # a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot, \
    #     a_hcap_tot, h_hcap_tot, a_ou, h_ou = ["NaN" for _ in range(12)]

    for market in markets:
        market_desc = market.get('description')
        mkt_type = reduce_mkt_type(market_desc)
        if not mkt_type:
            continue

        period_desc, abbrv, live = mkt_period_info(market)

        desc = clean_desc(market_desc)
        mkt_key = desc + '_' + mkt_type + '_' + abbrv

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
    period_desc, abbrv, live = mkt_period_info(market)
    outcomes = market.get('outcomes')

    market_desc = market.get('description')
    mkt_type = reduce_mkt_type(market_desc)

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

    ret.append(live)
    return ret


def mkt_period_info(market):
    '''
    returns the desc, abbrev, and live 
    '''
    period = market.get('period')
    to_grab = ['description', 'abbreviation', 'live']
    period_info = parse_json(period, to_grab, 'list')
    return period_info


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
    {game_id : quarter, secs, a_pts, h_pts, status}
    '''
    ids = get_ids(events)
    links = [bm.BOV_SCORES_URL + game_id for game_id in ids]
    if session:
        raw = g.async_req(links, output='dict',
                           key='eventId', session=session)
    else:
        raw = g.req_json(links)
    scores_dict = {g_id: score(j) for g_id, j in raw.items()}
    return scores_dict


def score(json_data):
    '''
    given json data for a game_id, returns the score data of the game
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


def filtered_links(sports, verbose=False):
    # append market filter to each url
    sfx = '?marketFilterId=def&lang=en'
    links = [bm.BOV_URL + u.match_sport_str(s) + sfx for s in sports]
    if verbose:
        print(f'bov_links: {links}')
    return links


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


def events_from_json(json_dict):
    '''
    simply accesses the events in a single json
    '''
    if not json_dict:
        return []
    events = json_dict[0]['events']
    return events


def match_sport_str(sport='mlb'):
    '''
    maps string to the url suffix to bovada api
    give sport='all' to get a list of all  
    '''
    try:
        sport = m.SPORT_TO_SUFFIX[sport]
    except KeyError:
        print('forcing nfl')
        sport = m.SPORT_TO_SUFFIX['nfl']
    return sport


def bov_all_dict():
    '''

    '''
    all_dict = {}
    req = r.get('https: // www.bovada.lv/services/sports /'
                'event/v2/events/A/description/basketball/nba').json()
    es = req[0].get('events')
    for event in es:
        desc = event.get('description')
        # print(f'desc: {desc}')
        if not desc:
            continue
        event_dict = u.parse_display_groups(event)
        cleaned = u.clean_desc(desc)
        all_dict[cleaned] = event_dict
    # print(f'all_dict: {all_dict}')
    return all_dict


if __name__ == "__main__":

    ll.main()
