import requests as r
import bs4

import time

DELAY=0.05
TIME_GAME_TUP=('a' , 'name', '&lpos=nfl:schedule:time')
TIME_GAME_TUP=('a' , 'name', '&lpos=nfl:schedule:score')
TIME_GAME_TUP=('a' , 'name', '&lpos=nfl:schedule:live')

def espn_boxlinks(ids):
    if not ids:
        ids = get_espn_ids()
    url = 'https://www.espn.com/nfl/boxscore?gameId='
    links = [url + id for id in ids]
    return links


def espn_time_ids(page=None):
    if not page:
        page = espn_schedule()
    unparsed_ids = page.find_all('a', {'name' : '&lpos=nfl:schedule:time'})
    time_ids = parse_ids(unparsed_ids)
    return time_ids


def espn_score_ids(page=None):
    if not page:
        page = espn_schedule()
    unparsed_ids = page.find_all('a', {'name' : '&lpos=nfl:schedule:score'})
    score_ids = parse_ids(unparsed_ids)
    return score_ids

def espn_live_ids(page=None):
    if not page:
        page = espn_schedule()
    unparsed_ids = page.find_all('a', {'name' : '&lpos=nfl:schedule:live'})
    live_ids = parse_live_ids(unparsed_ids)
    return live_ids


def espn_live_links(page=None):
    if not page:
        page = espn_schedule()
    live_ids = espn_live_ids(page)
    links = espn_boxlinks(live_ids)
    return links


def espn_schedule():
    espn_id_link = 'https://www.espn.com/nfl/schedule'
    p = get_page(espn_id_link)
    return p


def get_espn_ids():
    p = espn_schedule()
    live_ids = espn_live_ids(p)
    score_ids = espn_score_ids(p)
    time_ids = espn_time_ids(p)
    ids =  live_ids + score_ids + time_ids
    return ids


def get_live_pages():
    ids = espn_live_ids()
    pages = []
    # add multithreading
    for id in ids:
        pages.append(get_page(id_to_boxlink(id)))
    return pages


def get_live_boxes(pages=None):
    if not pages:
        pages = get_live_pages()
    boxes = []
    for p in pages:
        boxes.append(get_boxscore(p))
    return boxes


def parse_live_id(tag):
    id = tag['href'].split('=')[-1]
    return id


def parse_live_ids(tags):
    ret = []
    for tag in tags:
        ret.append(parse_live_id(tag))
    return ret


def parse_ids(ids):
    parsed_ids = []
    if ids:
        for tag in ids:
            id = tag['href'].split('/')[-1]
            parsed_ids.append(id)
    return parsed_ids


def id_to_boxlink(id):
    return 'https://www.espn.com/nfl/boxscore?gameId=' + id


def espn_box_tds(tds):
    x = None
    data = []
    for td in tds:
        if x == 1:
            data.append(td.text)
        txt = td.text
        if txt == 'TEAM':
            x = 1
    return data


def espn_teamstats(page):
    a_newstats = []
    h_newstats = []
    # if table_index % 2 == 1, then home_team
    tables = page.find_all('div', {'class' : 'content desktop'})
    for i, table in enumerate(tables):
        header = table.find('thead')
        len_header = len(header.find_all('th')) - 1  # - 1 for 'TEAM'
        tds = table.find_all('td')
        data = espn_box_tds(tds)
        if i % 2 == 1:
            if len(data) == 0:
                h_newstat = ['NaN' for _ in range(len_header)]
            else:
                h_newstat = data
            h_newstats.append(h_newstat)
        else:
            if len(data) == 0:
                a_newstat = ['NaN' for _ in range(len_header)]
            else:
                a_newstat = data
            a_newstats.append(a_newstat)
    return a_newstats, h_newstats


def parse_teamstats(teamstats):
    a_newstats, h_newstats = teamstats
    real_stats = []
    for team_newstats in teamstats:
        for team_newstat in team_newstats:
            for stat in team_newstat:
                try:
                    real_stat = stat.text
                    print(real_stat)
                    if real_stat == 'TEAM':
                        continue
                except AttributeError:
                    real_stat = stat
                real_stats.append(real_stat)
    return real_stats


def espn_box_teamnames(page):
    # A @ H always
    teams = page.find_all('span', {'class' : 'short-name'})
    destinations = page.find_all('span', {'class' : 'long-name'})
    names = [team.text for team in teams]
    cities = [destination.text for destination in destinations]
    a_team, h_team = [dest + ' ' + name  for (dest, name) in zip(cities, names)]
    return a_team, h_team


def get_pages(links):
    if not links:
        links = espn_boxlinks()
    boxes = []
    for link in links:
        box = get_boxscore(link)
        boxes.append(box)
    return boxes


def get_page(link):
    req = r.get(link).text
    p = bs4.BeautifulSoup(req, 'html.parser')
    time.sleep(DELAY)
    return p


def get_boxscore(link='https://www.espn.com/nfl/boxscore?gameId=401127863'):
    page = get_page(link)
    a_team, h_team = espn_box_teamnames(page)
    team_stats = espn_teamstats(page)
    real_stats = parse_teamstats(team_stats)
    real_stats.extend([a_team, h_team])
    return real_stats

if __name__ == "__main__":
    real_stats = get_boxscore()
    print(real_stats)
