'''
uses the bovada api to get json data for odds and scores
'''
import requests as r
import sips.h.grab as g
from sips.macros import macros as m
from sips.macros import bov as bm
from sips.lines.bov.utils import bov_utils as u


def get_links(sports, all_mkts=True):
    if all_mkts:
        links = [bm.BOV_URL + u.match_sport_str(s) for s in sports]
    else:
        links = u.filtered_links(sports)
    return links


def sports_to_jsons(sports, all_mkts=True):
    links = get_links(sports, all_mkts=all_mkts)
    jsons = g.reqs_json(links)
    return jsons


def lines(sports, output='list', parse=True, all_mkts=True, verbose=False):
    '''
    returns either a dictionary or list
    dictionary - (game_id, row)
    '''
    jsons = sports_to_jsons(sports, all_mkts)
    events = u.events_from_jsons(jsons)

    if output == 'dict':
        data = u.dict_from_events(events, key='id', rows=parse)
    else:
        data = [u.parse_event(e) for e in events]

    return data


def main():
    data = lines(["nba"], output='dict')
    print(data)
    return data


if __name__ == '__main__':
    data = main()
