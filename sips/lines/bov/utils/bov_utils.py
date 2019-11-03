'''
utils functions for bov.py
'''

import time
import requests as r

import sips.h.fileio as io
import sips.h.grab as g
import sips.h.parse as p
from sips.macros import bov as bm
from sips.macros import macros as m
from sips.lines import lines as ll
from sips.lines.bov.utils import scores


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


def parse_display_groups(event):
    '''
    given an event, it will parse all of the displaygroups
    returns a dictionary with a key for every group and the value is the data of
    each market in that group
    '''
    groups = event['displayGroups']
    full_set = {}
    for group in groups:
        desc = group.get('description')
        if not desc:
            continue
        # cleaned = clean_desc(desc)
        data_dict = parse_display_group(group)
        full_set[desc] = data_dict

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


def get_links(sports, all_mkts=True):
    if all_mkts:
        links = [bm.BOV_URL + match_sport_str(s) for s in sports]
    else:
        links = filtered_links(sports)
    return links


def sports_to_jsons(sports, all_mkts=True):
    links = get_links(sports, all_mkts=all_mkts)
    jsons = g.reqs_json(links)
    return jsons


def sports_to_events(sports, all_mkts=False):
    jsons = sports_to_jsons(sports=sports, all_mkts=all_mkts)
    events = events_from_jsons(jsons)
    return events


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
    events = [parse_event(e) for j in jsons
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
    game_id, sport, live, num_markets, last_mod = p.parse_json(
        event, ['id', 'sport', 'live', 'numMarkets', 'lastModified'],
        output='list')
    a_team, h_team = teams(event)
    game_start_time = event.get('startTime')
    display_groups = event.get('displayGroups')
    markets = [dg.get('markets') for dg in display_groups] 

    events_all_mkts = parse_display_groups(event)
    game_lines = events_all_mkts.get('Game Lines')

    if not game_lines:
        a_ps, h_ps, a_hcap, h_hcap, ps_M_live, a_team, a_ml, h_team, \
            h_ml, ml_M_live, a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, \
            h_ou, tot_M_live = [None for _ in range(17)]
    else:
        ml_M = game_lines.get('Moneyline_ml_M')
        ps_M = game_lines.get('Point Spread_ps_M')
        tot_M = game_lines.get('Total_tot_M')

        # to fix
        if ml_M:
            a_team, a_ml, h_team, h_ml, ml_M_live = ml_M
        else:
            a_team, a_ml, h_team, h_ml, ml_M_live = [None for _ in range(5)]
        if ps_M:
            a_ps, h_ps, a_hcap, h_hcap, ps_M_live = ps_M
        else:
            a_ps, h_ps, a_hcap, h_hcap, ps_M_live = [None for _ in range(5)]
        if tot_M:
            a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou, tot_M_live = tot_M
        else:
            a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou, tot_M_live = [
                None for _ in range(7)]


    score_url = bm.BOV_SCORES_URL + game_id
    score_data = g.req_json(score_url)
    quarter, secs, a_pts, h_pts, status = scores.score(score_data)

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
    keys of all_markets are: mkt_desc + reduced mkt type + period abbrv

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
        # print(f'market_desc: {market_desc}')
        mkt_type = reduce_mkt_type(market_desc)
        if not mkt_type or market_desc == 'Futures':
            continue

        period_desc, abbrv, live = mkt_period_info(market)

        # desc = clean_desc(market_desc)
        mkt_key = market_desc + '_' + mkt_type + '_' + abbrv

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
    period_info = p.parse_json(period, to_grab, 'list')
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
            a_ps, a_hcap = p.parse_json(price, TO_GRAB['ps'], 'list')
        else:
            h_ps, h_hcap = p.parse_json(price, TO_GRAB['ps'], 'list')

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
        # cleaned = clean_desc(desc)
        price = oc.get('price')
        american = price['american']
        mls[competitor_id] = [desc, american]
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
    a_tot, a_hcap_tot = p.parse_json(a_price, TO_GRAB['tot'], 'list')

    h_outcome = outcomes[1]
    h_ou = h_outcome.get('type')
    h_price = h_outcome.get('price')
    h_tot, h_hcap_tot = p.parse_json(h_price, TO_GRAB['tot'], 'list')

    return [a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou]


def competitors(competitors, verbose=False):
    '''
    keys: 'home' (bool), 'id' (str), 'name' (str)
    '''
    data = [p.parse_json(t, TO_GRAB['competitors']) for t in competitors]
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


def bov_team_ids(event):
    '''
    get competitor ids
    '''
    comps = event.get('competitors')
    if not comps:
        return 'NaN', 'NaN'
    teams = competitors(comps)
    a_id, h_id = [team['id'] for team in teams]
    return a_id, h_id


def filtered_links(sports, verbose=False):
    # append market filter to each url
    sfx = '?marketFilterId=def&lang=en'
    links = [bm.BOV_URL + match_sport_str(s) + sfx for s in sports]
    if verbose:
        print(f'bov_links: {links}')
    return links


def get_ids(events):
    # returns the ids for all bov events given
    ids = [e.get('id') for e in events if e.get('id')]
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
        event_dict = parse_display_groups(event)
        # cleaned = clean_desc(desc)
        all_dict[desc] = event_dict
    # print(f'all_dict: {all_dict}')
    return all_dict


if __name__ == "__main__":

    ll.main()
