'''
uses the bovada api to get json data for odds and scores
'''
import os

import time
import json

import requests as r
from requests_futures.sessions import FuturesSession

import sips.h.macros as m
from sips.lines.bov.utils import bov_utils as utils


def get_events(sports=['nba', 'mlb', 'nfl'], output='list'):
    '''
    gets all events for all the sports specified in macros.py
    output: either 'list' or 'dict', where each key is the game_id
    '''
    jsons = [utils.req_json(utils.match_sport_str(sport)) for sport in sports]
    events = utils.list_from_jsons(jsons)

    if output == 'dict':
        events = utils.dict_from_events(events)

    return events


def lines(sports, output='list', verbose=False, fixlines=True):
    '''
    returns either a dictionary or list
    dictionary - (game_id, row)
    '''
    if not sports:
        print(f'sports is None')
        return
        
    if fixlines:
        sfx = '?marketFilterId=def&lang=en'
        links = [utils.match_sport_str(s) + sfx for s in sports]
        jsons = [utils.req_json(l) for l in links]
    else:        
        links = [utils.match_sport_str(s) for s in sports]
        jsons = [utils.req_json(l) for l in links]
        

    jsons = [json for json in jsons if json]

    events = utils.list_from_jsons(jsons)

    if output == 'dict':
        data = utils.dict_from_events(events, rows=True)
    else:
        data = [utils.parse_event(e) for e in events]

    if verbose:
        print(f'rows: {data}')
        
    return data


def main():
    data = lines(["nba"], output='list')
    print(data)
    return data


if __name__ == '__main__':
    main()
