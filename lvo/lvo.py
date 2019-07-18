import requests as r
import bs4

'''
Schema
root -> each sport (global and lv)
each sport -> # TODO each game

currently have code to get every link, but there is like 90% redundancy

# todos
store list of sports and extensions for url construction instead of bs4 retrieval



'''
class LV:
    def __init__(self):
        self.url = 'http://www.vegasinsider.com'
        self.get_soup_urls()

    def get_soup_urls(self):
        p = r.get(self.url + '/odds/').text
        self.soup = bs4.BeautifulSoup(p, 'html.parser')
        gl_odds_urls = self.soup.find('img', {'alt' : 'Free Global Sportsbook Betting Odds and Lines'}).find_parent('tbody')
        lv_odds_urls = self.soup.find('img', {'alt' : 'Free Las Vegas Sportsbook Betting Odds and Lines'}).find_parent('tbody')
        self.odds_links = [link['href'] for link in gl_odds_urls.find_all('a') + lv_odds_urls.find_all('a')]

    def test_sport(self):
        for odd_link in self.odds_links:
            pages = [r.get(root + ext).text for ext in exts]
            soups = [bs4.BeautifulSoup(p, 'html.parser') for p in pages]
            links = []
            for dt in data_tables:
                links += [link['href'] for link in dt.find_all('a', {'class' : 'cellTextNorm'})]
