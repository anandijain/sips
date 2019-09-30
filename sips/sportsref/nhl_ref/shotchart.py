import bs4
import requests as r


import pandas as pd
import numpy as np

import sips.sportsref.h.parse as parse

link = 'https://www.hockey-reference.com/boxscores/201904100NYI.html'
root = 'https://www.hockey-reference.com'


def get_divs_list(charts):
    chart_list = []
    for chart in charts:
        divs = chart.find_all('div')
        chart_list.append(divs)
    # print(f'len chart_list: {len(chart_list)}')
    return chart_list

def grab_charts(link):
    # given a link to a hockey-refference boxscore, returns div, class: shotchart
    req = r.get(link).text
    p = bs4.BeautifulSoup(req, 'html.parser')
    cs = parse.comments(p)
    shotchart_comment = cs[22]
    chart_html = bs4.BeautifulSoup(shotchart_comment, 'html.parser')
    charts = chart_html.find_all('div', {'class': 'shotchart'})
    return charts


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


def div_dict_row(div, dict, game_id):
    x, y = div_coords(div)
    # print(x, y)
    type = shot_type(div['class'])
    title, player = shot_title(div['title'])

    dict['game_id'].append(game_id)
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


def parse_chart(div_list, game_id):
    # top, left, shot_type, title, player
    coords = []
    dict = {'game_id': [], 'x': [], 'y': [], 'shot_type': [], 'title': [], 'player': []}
    for div in div_list:
        try:
            cs = div_coords(div)
        except KeyError:
            continue
        dict = div_dict_row(div, dict, game_id)

    return dict



def dict_to_numpy(dict):
    df = pd.DataFrame(dict).values
    return df

def dicts_to_numpy(dicts):
    dfs = []
    for d in dicts:
        df = pd.DataFrame(d).values
        dfs.append(df)

    return dfs

def serialize():
    pass

def grab(link, fn=None):
    game_id = sfx_to_gameid(link)
    charts = grab_charts(link=link)
    chart_list = get_divs_list(charts)
    dict_list = []
    for chart in chart_list:
        dict = parse_chart(chart, game_id)
        if len(dict['x']) > 0:
            dict_list.append(dict)

    if fn:
        append_csv_dicts(fn, dict_list)

    return dict_list

def append_csv_dicts(fn, dicts):
    with open(fn, 'a') as f:
        for dict in dicts:
            df = pd.DataFrame(dict)
            df.to_csv(f, header=False)

def sfx_to_gameid(sfx):
    '''
    /boxscores/201810040OTT.html to 201810040OTT
    '''
    return sfx.split('/')[-1].split('.')[0]

def boxlinks():
    df = pd.read_csv('nhl_boxlinks.csv')
    sfxs = df.iloc[:, 1].values
    return sfxs


def main():
    '''
    outputs one large DataFrame
    game_id, x, y, type, outcome, player
    '''

    write_path = "./data/shots.csv"
    init_file(fn=write_path)

    sfxs = boxlinks()

    for i, sfx in enumerate(sfxs):
        link = root + sfx
        grab(link, fn=write_path)

        if i % 200 == 0:
            print(f'cur_game: {sfx_to_gameid(sfx)}')

def init_file(fn='shots.csv'):
    columns = ['game_id', 'x', 'y', 'type', 'outcome', 'player']
    f = open(fn, 'w+')
    write_header(f, columns)
    f.close()


# takes in list of strings
def write_header(file, columns):
	length = len(columns)
	for i, col in enumerate(columns):

		file.write(col)

		if i == length - 1:
			file.write('\n')
		else:
			file.write(',')


if __name__ == '__main__':
    main()
    # dl = grab(link=link)
    # dl
    # dl[0]
    # pass
