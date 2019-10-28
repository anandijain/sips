import requests as r

import sips.h.new_macros
from sips.lines.bov.utils import bov_utils as utils


def main():
    '''

    '''
    all_dict = {}
    req = r.get(
        'https://www.bovada.lv/services/sports/\
        event/v2/events/A/description/basketball/nba').json()
    es = req[0].get('events')
    for event in es:
        desc = event.get('description')
        # print(f'desc: {desc}')
        if not desc:
            continue
        event_dict = utils.parse_display_groups(event)
        cleaned = utils.clean_desc(desc)
        all_dict[cleaned] = event_dict
    # print(f'all_dict: {all_dict}')
    return all_dict


def reduce_group_desc(display_group):
    '''
    bins the display groups based on their general content
    as there are many repeats, (eg 'Alternate Lines' and 'Game Lines)
    '''
    reduced_group_type = None
    desc = display_group.get('description')
    if 'Props' in desc:
        reduced_group_type = 'props'
    elif 'Lines' in desc:
        reduced_group_type = 'lines'
    else:
        print('group type not supported yet')
        return None
    return reduced_group_type


if __name__ == "__main__":
    main()
