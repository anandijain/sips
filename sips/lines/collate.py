import time
import json

import requests as r

import sips.lines.bov as bov
from sips.lines.espn import eb, api


def get_and_compare():
    bov, api_evs, box_evs = get_events()
    rows = match_events(bov, espn)
    # rows = match_lines_boxes(bov, box_evs)
    return rows


def box_lines_comp():
    lines = bov.lines()
    boxes = eb.boxscores()
    rows = match_lines_boxes(lines, boxes)
    return rows


def get_events():
    bov_events = bov.events()
    espn_events = api.events()
    espn_boxes = eb.boxscores()
    return bov_events, espn_events, espn_boxes


def match_events(bov_events, espn_events):
    num_matched = 0
    rows = []
    eteams = None
    for event in bov_events:
        bteams = bov.teams(event)
        print(f'bteams: {bteams}')
        print(f'eteams: {eteams}')
        for espn_event in espn_events:
            eteams = api.teams(espn_event)
            if list(bteams) == list(eteams):
                print(f'games matched: {bteams}')
                line = bov.parse_event(event)
                espn_data = api.parse_event(espn_event)
                row = line + espn_data
                rows.append(row)
                num_matched += 1
    print(f'len(bov_events): {len(bov_events)}\nlen(espn_events): {len(espn_events)}')
    print(f'num_matched: {num_matched}')
    return rows


def match_lines_boxes(lines, boxes):
    num_matched = 0
    rows = []
    eteams = None
    for line in lines:
        bteams = bov.teams_from_line(line)
        # print(f'bteams: {bteams}')
        # print(f'eteams: {eteams}')
        for boxscore in boxes:
            eteams = boxscore[-2:]
            if list(bteams) == list(eteams):
                print(f'games matched: {bteams, eteams}')
                row = line + boxscore
                rows.append(row)
                num_matched += 1
    print(f'len(bov_events): {len(lines)}\nlen(espn_events): {len(boxes)}')
    print(f'num_matched: {num_matched}')
    return rows


def main():
    rows = box_lines_comp()
    print(rows)
    return rows


if __name__ == "__main__":
    rows = main()
