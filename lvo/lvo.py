import time

import requests as r
import bs4


'''
Schema
1. get sport names
self.sports = [string list]
2. for each sport go to scoreboard
self.root + sport + /scoreboard

lines = mlbs.find_all('a', href=True, text='Line Movements')


# todos
store list of sports and extensions for url construction instead of bs4 retrieval
currently have code to get every link, but there is like 90% redundancy

example line url :
http://www.vegasinsider.com/wnba/odds/global/line-movement/mystics-@-fever.cfm/date/07-19-19
split('/')
'''

headers = {'User-Agent': 'Mozilla/5.0'}

class LV:
    def __init__(self):
        self.url = 'http://www.vegasinsider.com'

        self.sports = \
            ['nfl', 'nba', 'nhl', 'mlb', 'college-football', 'college-basketball', 'auto-racing', 'golf'
            'horse-racing', 'soccer', 'boxing', 'wnba', 'afl', 'poker', 'cfl']

        self.odds_urls = []  # list of urls
        self.odds_tables = []

        self.get_odds_urls()

    def get_odds_urls(self):
        for sport in self.sports:
            sb_url = self.url + '/' + sport + '/scoreboard/'
            try:
                scoreboard = req_soup(sb_url)
            except Exception:
                continue
            lines = scoreboard.find_all('a', href=True, text='Line Movements')
            if len(lines) == 0:
                continue
            self.odds_urls += [line['href'] for line in lines]

        self.odds_urls += [global_from_lv(url) for url in self.odds_urls]
        # gl_odds_urls = self.soup.find('img', {'alt' : 'Free Global Sportsbook Betting Odds and Lines'}).find_parent('tbody')
        # lv_odds_urls = self.soup.find('img', {'alt' : 'Free Las Vegas Sportsbook Betting Odds and Lines'}).find_parent('tbody')
        # self.odds_links = [link['href'] for link in gl_odds_urls.find_all('a') + lv_odds_urls.find_all('a')]

    def get_game_tables(self, url):
        lines = req_soup(self.url + url)

        self.odds_tables += lines.find_all('table', {'class' : 'rt_railbox_border2'})

    def test_sport(self):
        for odd_link in self.odds_links:
            pages = [r.get(root + ext).text for ext in exts]
            soups = [bs4.BeautifulSoup(p, 'html.parser') for p in pages]
            links = []
            for dt in data_tables:
                links += [link['href'] for link in dt.find_all('a', {'class' : 'cellTextNorm'})]

def req_soup(url):
    page = r.get(url, headers=headers).text
    soup = bs4.BeautifulSoup(page, 'html.parser')
    time.sleep(.1)
    return soup


def global_from_lv(url):
    global_url = ''
    spl = url.split('/')
    for item in spl:
         if spl.index(item) == 5:
                 item = 'global'
         item += '/'
         global_url += item
    return global_url
