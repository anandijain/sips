'''
uses the bovada api to get json data for odds and scores
'''
import requests as r
import sips.h.macros as m
import sips.h.openers as o
from sips.lines.bov.utils import bov_utils as u


def get_events(sports=['nfl'], output='list', session=None):
    '''
    gets all events for all the sports specified in macros.py
    output: either 'list' or 'dict', where each key is the game_id
    '''
    links = fix_links(sports)
    jsons = [u.req_json(l) for l in links]
    # jsons = o.async_req(links, session=session)
    events = u.list_from_jsons(jsons)

    if output == 'dict':
        events = u.dict_from_events(events)

    return events


def fix_links(sports):
    # append market filter to each url
    sfx = '?marketFilterId=def&lang=en'
    links = [m.BOV_URL + u.match_sport_str(s) + sfx for s in sports]
    return links


def lines(sports, output='list', verbose=False, fixlines=True,
          session=None, espn=False):
    '''
    returns either a dictionary or list
    dictionary - (game_id, row)
    '''
    if not sports:
        print(f'sports is None')
        return

    if fixlines:
        links = fix_links(sports)
    else:
        links = [m.BOV_URL + u.match_sport_str(s) for s in sports]

    jsons = o.async_req(links, session=session)
    events = u.list_from_jsons(jsons)

    if output == 'dict':
        lines = u.dict_from_events(events, key='id', rows=True,
                                   grab_score=False)
        scores = u.get_scores(events, session=session)
        data = u.merge_lines_scores(lines, scores)
    else:
        data = [u.parse_event(e, grab_score=True) for e in events]

    if verbose:
        print(f'lines: {data}')

    return data


def main():
    data = lines(["nba"], output='list')
    print(data)
    return data


if __name__ == '__main__':
    main()
