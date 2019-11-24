import time

from sips.lines.bov import bov
from sips.lines.espn import espn_box as eb
from sips.lines.espn import espn_api as api

from sips.lines.bov import bov
from sips.lines.bov.utils import bov_utils as u


def to_dict(lists):
    """
    convert lists to a dictionary where the index-th elt is the key
    not needed?
    """
    return {l[i]: l for i, l in enumerate(lists)}


def get_and_compare(sports=["football/nfl", "basketball/nba"], output="dict"):
    """
    access the espn boxscores and api and merge with bovada lines 
    """
    bov, api_evs, boxes = get_events(sports=sports)
    api_lines_rows = match_api_lines(bov, api_evs)
    rows = match_lines_boxes(api_lines_rows, boxes, output="dict")
    return rows


def get_events(sports=["football/nfl"], verbose=False):
    bov_events = u.sports_to_events(sports=sports, all_mkts=False)
    espn_events = api.req_events(sports=sports)
    espn_boxes = eb.boxscores(sports=sports)
    if verbose:
        print(f"bov_events: {bov_events}")
        print(f"espn_events: {espn_events}")
        print(f"espn_boxes: {espn_boxes}")
    return bov_events, espn_events, espn_boxes


def match_api_lines(bov_events, espn_events, output="list"):
    num_matched = 0
    rows = []
    eteams = None
    for event in bov_events:
        bteams = u.teams(event)
        for espn_event in espn_events:
            eteams = api.teams(espn_event)
            if sorted(list(bteams)) == sorted(list(eteams)):
                print(f"games matched: {bteams}")
                line = u.parse_event(event)
                espn_data = api.parse_event(espn_event)
                row = line + espn_data
                rows.append(row)
                num_matched += 1
    print(
        f"len(bov_events): {len(bov_events)}\n" f"len(espn_events): {len(espn_events)}"
    )
    print(f"num_matched: {num_matched}")
    ret = []
    for row in rows:
        row = [elt for elt in row if elt != "--"]
        ret.append(row)
    rows = ret

    if output == "dict":
        rows = to_dict(rows)
    return rows


def match_lines_boxes(lines, boxes, output="dict", verbose=True):
    num_matched = 0
    rows = []
    eteams = None
    for line in lines:
        bteams = line[2:4]
        if not bteams:
            print(f"bskip: {line}")
            continue
        for boxscore in boxes:
            eteams = boxscore[-2:]
            if not eteams:
                print(f"eskip: {boxscore}")
                continue
            try:
                teams_same = sorted(list(bteams)) == sorted(list(eteams))
            except TypeError:
                continue
            if teams_same:
                if verbose:
                    print(f"games matched: {bteams, eteams}")
                row = line + boxscore
                rows.append(row)
                num_matched += 1

    if output == "dict":
        rows = {row[1]: row for row in rows}

    if verbose:
        print(f"len(bov_events): {len(lines)}\nlen(espn_events): {len(boxes)}")
        print(f"num_matched: {num_matched}")
    return rows


def box_lines_comp(sports=["football/nfl"], output="dict"):
    lines = bov.lines(sports, output="list")
    boxes = eb.boxscores(sports=sports)
    rows = match_lines_boxes(lines, boxes, output=output)
    return rows


def main():
    start = time.time()
    rows = get_and_compare()
    end = time.time()
    delta = end - start
    print(f"delta: {delta}")
    print(f"all rows: {rows}")
    return rows


if __name__ == "__main__":
    rows = main()
