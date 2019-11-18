'''
 mainly bs4 utilities for grabbing data from sites:
'''
import bs4
import pandas as pd

import numpy as np


def read_html_with_links(html_table):
    thead = html_table.thead


def parse_json(json, keys, output='dict'):
    '''
    input: dictionary and list of strings
    returns dict

    if the key does not exist in the json
    the key is still created with None as the value
    '''
    data = {}
    json_keys = json.keys()
    for j_key in json_keys:
        if j_key in keys:
            d = json.get(j_key)
            data[j_key] = d

    if output == 'list':
        return list(data.values())
    elif output == 'dict':
        return data
    else:
        return None


def comments(page):
    # finds all of the bs4.Comments for a page
    comments = page.findAll(text=lambda text: isinstance(text, bs4.Comment))
    return comments


def get_table(page, table_id):
    # given bs4 page and table id, finds table using bs4. returns soup table
    table = page.find('table', {'id': table_id})
    return table


def columns_from_table(table, attr=None):
    '''
    given a bs4 table, gives string names of headers in list
    can provide arg attr that can give you a specific tag attribute
    '''
    thead = table['thead']
    if not thead:
        print('table has no headers')
        return
    headers = thead.find_all('th')
    if not attr:
        columns = [h.text for h in headers]
    else:
        columns = [h[attr] for h in headers]
    return columns


def parse_table(table, tag='th'):
    '''

    '''
    tbody = table.tbody
    rows = tbody.find_all('tr')
    data_rows = [[] for i in range(len(rows))]
    for row in rows:
        row_class = row.get('class')
        if row_class == 'spacer':
            continue
        # print(row.text)
        row_data = []
        things = row.find_all(tag)
        for thing in things:
            row_data.append(thing)
            # print(thing.text)
        data_rows.append(row_data)
    return data_rows


def write_table(table, fn, tag='th'):
    '''

    '''
    try:
        tbody = table.tbody
    except AttributeError:
        return
    try:
        file = open('.' + '/data/' + fn + '.csv', 'w')
    except FileExistsError:
        print('skip')
        return

    thead = table.thead
    columns_row = thead.tr
    col_items = columns_row.find_all('th')
    for i, col in enumerate(col_items):

        file.write(col.text)

        if i == len(col_items) - 1:
            file.write('\n')
        else:
            file.write(',')

    rows = tbody.find_all('tr')
    for row in rows:
        row_class = row.get('class')
        if row_class is None:  # when the row class is none it is a data row
            row_data = row.find_all(tag)
            for i, data_pt in enumerate(row_data):
                file.write(data_pt.text)

                if i == len(row_data) - 1:
                    file.write('\n')
                else:
                    file.write(',')

    print('{} written to {}'.format(fn, './data/'))
    file.close()


def serialize_row(row, teams_dict, statuses_dict):
    '''
    going to take in something like this:
    ['FOOT', 5741304, 'Pittsburgh Steelers', 'Cleveland Browns', 1573540736617, 28,
    False, '0', '-1', '0', '0', 'PRE_GAME', '2.5', '-2.5', '-105', '-115', '+125',
    '-145', '40.0', '40.0', '-110', '-110', 'O', 'U', 1573780800000]
    and return a np array
    '''
    ret = []
    row = list(row)
    teams = row[2:4]

    for t in teams:
        hot_teams = teams_dict[t]
        ret += hot_teams

    ret += row[4:6]

    if row[6]:
        ret += [1, 0]
    else:
        ret += [0, 1]

    ret += [row_ml(ml) for ml in row[7:11]]

    row_status = row[11]
    hot_status = statuses_dict[row_status]
    ret += hot_status
    mls = [row_ml(ml) for ml in row[12:22]]
    ret += mls
    final = np.array(ret, dtype=np.float32)
    return final


def row_ml(ml):
    '''
    given a list of unparsed moneylines (eg can be 'EVEN' and None)
    edit the values such that 'EVEN' -> 100 and None -> -1
    typical order of list is [a0, h0, a1, h1]
    '''
    if ml == 'EVEN':
        ret = 100
    elif ml == None:
        ret = -1
    else:
        try:
            ret = float(ml)
        except:
            ret = -1
    return ret


if __name__ == "__main__":
    pass
