'''
uses the bovada api to get json data for odds and scores
'''
import requests as r
import sips.h.grab as g
from sips.macros import macros as m
from sips.macros import bov as bm
from sips.lines.bov.utils import bov_utils as u


def get_events(sports=['nfl'], output='list', all_mkts=True):
    '''
    gets all events for all the sports specified in macros.py
    output: either 'list' or 'dict', where each key is the game_id
    '''
    if all_mkts:
        links = [bm.BOV_URL + u.match_sport_str(s) for s in sports]
    else:
        links = u.filtered_links(sports)

    jsons = [g.req_json(l) for l in links]
        
    events = u.events_from_jsons(jsons)

    if output == 'dict':
        events = u.dict_from_events(events, rows=False)

    return events


def lines(sports, output='list', verbose=False, all_mkts=True, espn=False):
    '''
    returns either a dictionary or list
    dictionary - (game_id, row)
    '''

    events = get_events(sports=sports, output=output, all_mkts=all_mkts)

    if output == 'dict':
        data = u.dict_from_events(events, rows=True)
    else:
        data = [u.parse_event(e, all_mkts=all_mkts) for e in events]

    if verbose:
        print(f'lines: {data}')

    return data


def main():
    data = lines(["nba"], output='list')
    print(data)
    return data


if __name__ == '__main__':
    data = main()
