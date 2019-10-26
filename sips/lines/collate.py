import time
import json

import requests as r

from sips.lines.bov import bov
from sips.lines.espn import eb, api

from sips.lines.bov import bov
from sips.lines.bov.utils import bov_utils

from nfl_ref import full_package as fp


def get_and_compare():
    bov, api_evs, box_evs = get_events()
    rows = match_events(bov, api_evs)
    # lines_boxes = match_lines_boxes(bov, box_evs)
    # df = fp.scraper_main()
    # bov_df = pd.DataFrame(bov.lines(['nfl']))
    # ret = pd.merge(by='outer', on='')
    return rows


def box_lines_comp(sports=['nfl']):
    lines = bov.lines(sports, output='list')
    sport = sports[0]
    print(f'sports[0]: {sports[0]}')
    boxes = eb.boxscores(sport)
    rows = match_lines_boxes(lines, boxes)
    return rows


def get_events(sports=['nba'], verbose=True):
    bov_events = bov.get_events(sports=sports)
    espn_events = api.events(sport=sports[0])
    espn_boxes = eb.boxscores(sport=sports[0])
    if verbose:
        print(f'bov_events: {bov_events}')
        print(f'espn_events: {espn_events}')
        print(f'espn_boxes: {espn_boxes}')
    return bov_events, espn_events, espn_boxes


def match_events(bov_events, espn_events):
    num_matched = 0
    rows = []
    eteams = None
    for event in bov_events:
        bteams = bov_utils.teams(event)
        print(f'bteams: {bteams}')
        print(f'eteams: {eteams}')
        for espn_event in espn_events:
            eteams = api.teams(espn_event)
            if list(bteams) == list(eteams):
                print(f'games matched: {bteams}')
                line = bov_utils.parse_event(event)
                espn_data = api.parse_event(espn_event)
                row = line + espn_data
                rows.append(row)
                num_matched += 1
    print(f'len(bov_events): {len(bov_events)}\nlen(espn_events): {len(espn_events)}')
    print(f'num_matched: {num_matched}')
    return rows


def match_lines_boxes(lines, boxes, verbose=True):
    num_matched = 0
    rows = []
    eteams = None
    for line in lines:
        bteams = sorted(bov_utils.teams_from_line(line))
        for boxscore in boxes:
            eteams = sorted(boxscore[-2:])
            print(f'bteams: {bteams}')
            print(f'eteams: {eteams}')
            if list(bteams) == list(eteams):
                if verbose:
                    print(f'games matched: {bteams, eteams}')
                row = line + boxscore
                rows.append(row)
                num_matched += 1
    if verbose:
        print(f'len(bov_events): {len(lines)}\nlen(espn_events): {len(boxes)}')
        print(f'num_matched: {num_matched}')
    return rows


def main():
    rows = box_lines_comp(['college-football'])
    print(rows)
    return rows


if __name__ == "__main__":
    rows = main()
