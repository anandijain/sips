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


def games(config_path):
    '''
    reading in config, we return the lines for the games specified
    '''
    with open(config_path) as config:
        conf = json.load(config)
    ids = conf['games'][0]['game_ids']
    lines_dict = lines(conf['sport'])


def get_events(sport, output='list'):
    '''
    gets all events for all the sports specified in macros.py
    output: either 'list' or 'dict', where each key is the game_id
    '''
    jsons = [utils.req_json(sport)]
    events = utils.list_from_jsons(jsons)

    if output == 'dict':
        events = utils.dict_from_events(events)

    return events


def lines(sport='nba', output='dict', verbose=False):
    '''
    returns either a dictionary or list
    dictionary - (game_id, row)
    '''
    sport = utils.match_sport_str(sport)
    json_data = utils.req_json(sport)
    events = utils.json_events(json_data)

    if output == 'dict':
        data = utils.dict_from_events(events, rows=True)
    else:
        data = [utils.parse_event(e) for e in events]

    if verbose:
        print(f'rows: {data}')
    return data


def main():
    data = lines()
    print(data)
    return data


if __name__ == '__main__':
    main()
