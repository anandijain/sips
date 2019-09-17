import requests as r
import pandas as pd

import bs4

url = "https://www.hockey-reference.com"

def get_page(url):
    req = r.get(url)
    p = bs4.BeautifulSoup(req.text, "html.parser")
    return p

def league_index():
    suffix = "/leagues/"
    p = get_page(url + suffix)
    t = p.find('table', {'id' : 'league_index'})
    return t

def find_in_table(t, tup):
    # tup is 3tuple with ('th', 'data-stat', 'season') for example
    selected = t.find_all(tup[0], {tup[1] : tup[2]})
    links = []
    for sel in selected:
        try:
            links.append(sel.a['href'])
        except TypeError:
            continue
    return links

def link_fix(link):
    split = link.split('.')
    split[0] += "_games." 
    ret = split[0] + split[1]
    return ret

def gamelinks_str_fix(links):
    ret = []
    for link in links:
        ret.append(link_fix(link))
    return ret

def season_boxlinks(season_url):
    find_tup = ('th', 'data-stat', 'date_game')
    p = get_page(season_url)
    tables = p.find_all('table')
    ret = []
    for table in tables:    
        ret += find_in_table(table, find_tup)
    return ret

def main():
    ret = []
    t = league_index()
    ls = find_in_table(t, ('th', 'data-stat', 'season'))
    ls = gamelinks_str_fix(ls)
    for l in ls[1:]:
        print(l)
        ret += season_boxlinks(url + l)

    return ret
