import requests as r
import bs4

import time

ESPN_ROOT='https://www.espn.com'

DELAY=0.05
TIME_GAME_TUP=('a' , 'name', '&lpos=nfl:schedule:time')
TIME_GAME_TUP=('a' , 'name', '&lpos=nfl:schedule:score')
TIME_GAME_TUP=('a' , 'name', '&lpos=nfl:schedule:live')


def get_sports():
    sports = 'nfl', 'mlb', 'nba', 'college-football', 'mens-college-basketball'
    return sports


def boxlinks(ids=None, sport='nfl'):
    if not ids:
        ids = get_ids(sport)
    url = ESPN_ROOT + '/' + sport + '/boxscore?gameId='
    links = [url + id for id in ids]
    return links


def time_ids(page=None, sport='nfl'):
    if not page:
        page = schedule(sport)
    unparsed_ids = page.find_all('a', {'name' : '&lpos=' + sport + ':schedule:time'})
    return unparsed_ids


def score_ids(page=None, sport='nfl'):
    if not page:
        page = schedule(sport)
    unparsed_ids = page.find_all('a', {'name' : '&lpos=' + sport + ':schedule:score'})
    return unparsed_ids


def live_ids(page=None, sport='nfl'):
    if not page:
        page = schedule(sport)
    unparsed_ids = page.find_all('a', {'name' : '&lpos=' + sport + ':schedule:live'})
    live_ids = parse_live_ids(unparsed_ids)
    return live_ids


def get_all_ids(page=None, sports=get_sports()):
    ids = {}
    for sport in sports:
        ids[sport] = get_ids(sport)
    return ids


def live_links(page=None, sport='nfl'):
    if not page:
        page = schedule(sport)
    ids = live_ids(page, sport)
    links = boxlinks(ids, sport)
    return links


def schedules(sports=['nfl']):
    if not sports:
        sports = get_sports()

    pages = []
    for sport in sports:
            espn_id_link = '/' + sport + '/schedule'
            p = get_page(espn_id_link)
            pages.append(p)
    return pages


def schedule(sport='nfl'):
    espn_id_link = ESPN_ROOT + '/' + sport + '/schedule'
    p = get_page(espn_id_link)
    return p


def get_ids(sport='nfl'):
    p = schedule(sport)
    ids_live = live_ids(p, sport)
    ids_score = score_ids(p, sport)
    ids_time = time_ids(p, sport)
    ids_parsed = parse_ids(ids_score + ids_time)
    ids =  ids_live + ids_parsed
    return ids


def get_live_pages():
    ids = live_ids()
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
    # print(tag)
    id = tag['href'].split('=')[-1]
    return id


def parse_live_ids(tags):
    ret = []
    if isinstance(tags, dict):
        for k,v in tags.items():
            v = parse_live_id(v)
        return items
    else:
        for tag in tags:
            ret.append(parse_live_id(tag))
    return ret


def tag_to_id(tag):
    id = tag['href'].split('/')[-1]
    return id


def parse_id_dict(tags_dict):
    for k, v in tags_dict.items():
        tags_dict[k] = [parse_live_id(tag) for tag in v]
    return tags_dict


def parse_ids(tags):
    parsed_ids = []
    if tags:
        if isinstance(tags, dict):
            return parse_id_dict(tags)
        for tag in tags:
            # print(tag)
            try:
                id = tag_to_id(tag)
            except TypeError:
                pass
            try:
                id = id.split('=')[-1]
            except TypeError:
                pass
            print(id)
            parsed_ids.append(id)
    else:
        return None
    return parsed_ids


def id_to_boxlink(id, sport='nfl'):
    return ESPN_ROOT + '/' + sport + '/boxscore?gameId=' + id


def box_tds(tds):
    x = None
    data = []
    for td in tds:
        if x == 1:
            data.append(td.text)
        txt = td.text
        if txt == 'TEAM':
            x = 1
    return data


def teamstats(page):
    a_newstats = []
    h_newstats = []
    # if table_index % 2 == 1, then home_team
    tables = page.find_all('div', {'class' : 'content desktop'})
    for i, table in enumerate(tables):
        header = table.find('thead')
        len_header = len(header.find_all('th')) - 1  # - 1 for 'TEAM'
        tds = table.find_all('td')
        data = box_tds(tds)
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


def box_teamnames(page):
    # A @ H always
    teams = page.find_all('span', {'class' : 'short-name'})
    destinations = page.find_all('span', {'class' : 'long-name'})
    names = [team.text for team in teams]
    cities = [destination.text for destination in destinations]
    a_team, h_team = [dest + ' ' + name  for (dest, name) in zip(cities, names)]
    return a_team, h_team


def get_page(link):
    req = r.get(link).text
    p = bs4.BeautifulSoup(req, 'html.parser')
    time.sleep(DELAY)
    return p


def boxscores(links=None):
    if not links:
        links = boxlinks()
    boxes = []
    for link in links:
        box = boxscore(link)
        boxes.append(box)
    return boxes


def boxscore(link=ESPN_ROOT + '/nfl/boxscore?gameId=401127863'):
    page = get_page(link)
    a_team, h_team = box_teamnames(page)
    team_stats = teamstats(page)
    real_stats = parse_teamstats(team_stats)
    real_stats.extend([a_team, h_team])
    return real_stats


if __name__ == "__main__":
    real_stats = boxscores()
    print(real_stats)
