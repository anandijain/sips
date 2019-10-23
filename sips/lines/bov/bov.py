'''
uses the bovada api to get json data for odds and scores
'''
import time
import json

import requests as r
from requests_futures.sessions import FuturesSession

import sips.h.macros as m
from sips.lines.bov.utils import bov_utils as utils


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
                ret.append(utils.parse_event(event))
    return ret


def all_events(output='list'):
    '''
    gets all events for all the sports specified in macros.py
    output: either 'list' or 'dict', where each key is the game_id
    '''
    json_events_lists = utils.async_req(m.build_urls())
    
    if output == 'list':
        events = utils.events_from_jsons(json_events_lists)
    elif output == 'dict':
        events = utils.event_dict_from_jsons(json_events_lists)
    else:
        raise NotImplementedError('this output type is not supported')    
    return events


def get_events(sport='mlb', verbose=False):
    '''
    given a specific sport, return the events
    '''
    sport = utils.match_sport_str(sport)
    json_data = utils.req_json(sport)
    events = utils.events_from_json(json_data)

    if verbose:
        print(json_data)

    return events


def lines(events=None, sport='mlb'):
    '''
    returns the lines for a given sport
    '''
    if not events:
        events = get_events(sport=sport)
    rows = [utils.parse_event(e) for e in events]
    return rows


def main():
    data = lines()
    print(data)
    return data


if __name__ == '__main__':
    main()
