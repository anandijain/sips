'''
uses the bovada api to get json data for odds and scores
'''
import requests as r
import sips.h.grab as g
from sips.macros import macros as m
from sips.macros import bov as bm
from sips.lines.bov.utils import bov_utils as u


def lines(sports, output='list', parse=True, all_mkts=False, verbose=False):
    '''
    returns either a dictionary or list
    dictionary - (game_id, row)
    '''
    events = u.sports_to_events(sports, all_mkts)

    if output == 'dict':
        data = u.dict_from_events(events, key='id', rows=parse)
    else:
        data = [u.parse_event(e) for e in events]

    return data


def main():
    data = lines(["nba"], output='dict')
    print(data)
    print(len(data))
    return data


if __name__ == '__main__':
    data = main()
