import bs4
import requests as r


import pandas as pd
import numpy as np

import sips.sportsref.h.parse as parse

link = 'https://www.hockey-reference.com/boxscores/201904100NYI.html'
root = 'https://www.hockey-reference.com'

def grab(link, fn=None):
    game_id = sfx_to_gameid(link)
    charts = grab_charts(link=link)
    divs = get_divs(charts)

    if len(divs) == 0:
        return 0

    rows = divs_to_arr(divs)


    df = cat_id(rows, game_id)
    # df.columns = ['i', 'x', 'y', 'type', 'outcome', 'player', 'game_id']

    if fn:
        append_csv(fn, df)

    return df

def get_divs(charts):
    all_divs = []
    for chart in charts:
        divs = chart.find_all('div')
        all_divs += divs
    # print(f'len chart_list: {len(chart_list)}')
    # divs = list_flatten(chart_list)
    return all_divs

def divs_to_arr(divs):
    rows = []
    for div in divs:
        try:
            dict = arr_row(div)
        except KeyError:
            continue
        rows.append(dict)
    return rows

def arr_row(div):
    # game_id, x, y, shot_type, title, player
    x, y = div_coords(div)
    type = shot_type(div['class'])
    title, player = shot_title(div['title'])
    return [x, y, type, title, player]

def cat_id(rows, id):
    df = pd.DataFrame(rows)
    df['game_id'] = id
    return df

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

def div_dict_row(div, dict):
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

def parse_chart(divs, game_id):
    # game_id, x, y, shot_type, title, player
    coords = []
    print(divs)
    print(len(divs))
    ids = [game_id for _ in range(len(divs['x']))]
    dict = {'game_id': ids, 'x': [], 'y': [], 'shot_type': [], 'title': [], 'player': []}
    for div in divs:
        try:
            cs = div_coords(div)
        except KeyError:
            continue
        dict = div_dict_row(div, dict)

    return dict

def append_csv(fn, df):
    with open(fn, 'a') as f:
        df.to_csv(f, header=False)

def list_flatten(l):
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def sfx_to_gameid(sfx):
    '''
    /boxscores/201810040OTT.html to 201810040OTT
    '''
    return sfx.split('/')[-1].split('.')[0]

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
    # for testing
    # links = np.random.permutation(boxlinks())
    meta_df = pd.DataFrame(columns=['game_id', 'num_rows'])
    for i, sfx in enumerate(links):
        link = root + sfx
        game_id = sfx_to_gameid(sfx)
        df = grab(link, fn=write_path)
        meta_df = meta_df.append({'game_id': game_id, 'num_rows': len(df)}, ignore_index=True)
        if i % 200 == 0:
            game_id = sfx_to_gameid(sfx)
            print(game_id)
    meta_df.to_csv('./data/meta_shots.csv')

def init_file(fn='shots.csv'):
    columns = ['i', 'x', 'y', 'type', 'outcome', 'player', 'game_id']
    f = open(fn, 'w+')
    write_header(f, columns)
    f.close()


# takes in list of strings
def write_header(file, columns):
	length = len(columns)
	for i, col in enumerate(columns):

		file.write(col)

		if i >= length - 1:
			file.write('\n')
		else:
			file.write(',')


if __name__ == '__main__':
    main()
    # dl = grab(link=link)
    # dl
    # dl[0]
    # pass
