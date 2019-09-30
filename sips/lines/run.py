import time
import json

import requests as r

from sips.lines.espn_box import get_boxscore

import sips.lines.bov as bov
import sips.lines.espn_api as espn_api
import sips.lines.espn_box as espn_box



def get_and_compare():
    bov, espn = get_events()
    rows = match_events(bov, espn)
    return rows

def get_events():
    bov_events = bov.get_bov_events()
    espn_events = espn_api.get_espn_events()
    espn_boxes= espn_box.get_boxscores()
    return bov_events, espn_events, espn_boxes

def match_events(bov_events, espn_events):
    num_matched = 0
    rows = []
    eteams = None
    for event in bov_events:
        bteams = bov.bov_teams(event)
        print(f'bteams: {bteams}')
        print(f'eteams: {eteams}')
        for espn_event in espn_events:
            eteams = espn_api.espn_teams(espn_event)
            if list(bteams) == list(eteams):
                print(f'games matched: {bteams}')
                line = bov.bov_line(event)
                espn_data = espn_api.parse_espn_event(espn_event)
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
        print(f'bteams: {bteams}')
        print(f'eteams: {eteams}')
        for boxscore in boxes:
            # print(boxscore)
            eteams = boxscore[-2:]
            if list(bteams) == list(eteams):
                print(f'games matched: {bteams}')
                row = line + boxscore
                rows.append(row)
                num_matched += 1
    print(f'len(bov_events): {len(lines)}\nlen(espn_events): {len(boxes)}')
    print(f'num_matched: {num_matched}')
    return rows

def main():
    rows = get_and_compare()
    # print(rows)
    return rows


if __name__ == "__main__":
    rows = main()
