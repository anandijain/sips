import requests as r
import sips.h.new_macros
def main():
    req = r.get(
        'https://www.bovada.lv/services/sports/event/v2/events/A/description/basketball/nba').json()
    es = req[0].get('events')
    e = es[0]
    row = display_groups(e)

def parse_display_groups(event):
    '''
    important:
        - 'Game Lines': (periods: 1H, 2H and Match for each)
            - 'Moneyline' 
            - 'Point Spread'
            - 'Total'
            are the three main categories, but can be for any period 
        - 'Period/Alternate Lines'
        - 'Game Props'

        
        'Futures'
        'Alternate Lines'
    
    secondary:
        - '
        - 'Score Props'
        - 'Receiving Props'
        - 'Quarterback Props'
        - 'Rushing Props'
        - 'Touchdown Props'
        - 'Defense/Special Teams Props'
        - 'Punt Props'
        - 'Sack Props'
        - 'Defensive Player Props'
        - 'Field Goal Props
    '''

    groups = event['displayGroups']
    full_set = {}
    for group in groups:
        data_dict = parse_display_group(group)
        full_set.update(data_dict)

    return full_set


def reduce_mkt_type(market):
    desc = market.get('description')
    if not desc:
        print('no desc in mkt grab')
        return None
            
    MKT_TYPE = {
        'Point Spread': 'ps',
        'Runline': 'ps',
        'Moneyline': 'ml',
        'Total': 'tot'
    }

    return MKT_TYPE[desc]


def parse_lines_group(group):
    '''

    '''
    markets = group.get('markets')
    if not markets:
        print('no markets')
        return None
    for market in markets:
        data_dict = parse_market(market)

def parse_display_group(display_group):
    group_type = reduce_group_desc(display_group)

    return group_type
    
def reduce_group_desc(display_group):
    '''
    bins the display groups based on their general content
    as there are many repeats, (eg 'Alternate Lines' and 'Game Lines)
    '''
    reduced_group_type = None
    desc = display_group.get('description')
    if 'Props' in desc:
        reduced_group_type = 'props'
    elif 'Lines' in desc:
        reduced_group_type = 'lines'
    else:
        print('group type not supported yet')
        return None
    return reduced_group_type


def parse_market(market):
    '''
    given: market in bovada sport json
    returns: dictionary w (field , field_value)
    '''
    data_dict = {}
    mkt_type = reduce_mkt_type(market)
    if not mkt_type:
        return 'mkt type not supported'
    elif mkt_type == 'ps':
        data = parse_spread(market)
    elif mkt_type == 'ml':
        data = parse_moneyline(market)
    elif mkt_type == 'tot':
        data = parse_total(market)
    data_dict.update(data)
    return data_dict

def parse_moneyline(market):
    data_dict = {}
    outcomes = market.get('outcomes')
    if not outcomes:
        print('market has no outcomes data')
        return None
    period = market.get('period')
    period_keys = ['abbreviation', 'live']
    period_dict = parse_json(period, period_keys)
    data.update(period_dict)
    for outcome in outcomes:
        outcome_data = parse_json()

    return data

def parse_spread(market):
    return data

def parse_total(market):
    return data

def parse_json(json, keys):
    '''
    input: dictionary and list of strings
    returns dict
    '''
    data = {}
    json_keys = json.keys()
    for j_key in json_keys:
        if j_key in keys:
            d = json.get(k)    
            data[k] = d
    return data

if __name__ == "__main__":
    pass
    
