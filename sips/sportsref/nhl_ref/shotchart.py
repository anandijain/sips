import bs4
import requests as r

import pandas as pd
import numpy as np

import sips.sportsref.h.parse as parse

link = 'https://www.hockey-reference.com/boxscores/201904100NYI.html'
root = 'https://www.hockey-reference.com'


def grab_charts(link):
    # given a link to a hockey-refference boxscore, returns div, class: shotchart
    req = r.get(link).text
    p = bs4.BeautifulSoup(req, 'html.parser')
    cs = parse.comments(p)
    shotchart_comment = cs[22]
    chart_html = bs4.BeautifulSoup(shotchart_comment, 'html.parser')
    charts = chart_html.find_all('div', {'class': 'shotchart'})
    return charts


def grab_shot_divs(chart):
    # must be a shotchart div
    return chart.find_all('div')

def shot_title(title):
    shot_outcome, player = title.split(' - ')
    return shot_outcome, player

def shot_type(t):
    ret = None
    if len(t) > 1:
        t = ' '.join(t)
        ret = t
    else:
        ret = t[0]
    return ret


def div_dict(div, dict=None):
    if not dict:
        dict = {'x': [], 'y': [], 'shot_type': [], 'title': [], 'player': []}
    x, y = div_coords(div)
    # print(x, y)
    type = shot_type(div['class'])
    title, player = shot_title(div['title'])

    dict['x'].append(x)
    dict['y'].append(y)
    dict['shot_type'].append(type)

    dict['title'].append(title)
    dict['player'].append(player)

    return dict

def div_coords(div):
    # positions = top == y, left == x
    positions = div['style'].split(' ')
    x, y = positions[3], positions[1]
    x, y = [int(c.split('p')[0]) for c in (x, y)]
    return x, y

def parse_chart(div_list):
    # top, left, shot_type, title, player
    coords = []
    dict = {'x': [], 'y': [], 'shot_type': [], 'title': [], 'player': []}
    for div in div_list:
        try:
            cs = div_coords(div)
        except KeyError:
            continue
        dict = div_dict(div, dict)
        if cs:
            coords.append(cs)

    # print(coords)
    return coords, dict

def get_divs_list(charts):
    chart_list = []
    for chart in charts:
        divs = grab_shot_divs(chart)
        chart_list.append(divs)
    print(f'len chart_list: {len(chart_list)}')
    return chart_list

def dict_to_numpy(dict):
    df = pd.DataFrame(dl)
    return df.values


def serialize():
    pass

def grab(link):
    charts = grab_charts(link=link)
    chart_list = get_divs_list(charts)
    coords_list = []
    dict_list = []
    for chart in chart_list:
        coords, arr = parse_chart(chart)

        if coords:
            coords_list += coords
        if arr:
            dict_list.append(arr)

    # print(coords_list, dict_list)
    return coords_list, dict_list



def main(link=link):



if __name__ == '__main__':
    main(link=link)
