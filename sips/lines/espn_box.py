import requests as r
import bs4


def newstats_to_team_stats(newstats, page):
    a_newstats = []
    h_newstats = []
    for i, newstat in enumerate(newstats):
        if len(newstat) == 2:
            a_newstat = newstat[0].find_all('td')
            h_newstat = newstat[1].find_all('td')
            del a_newstat[0]
            del h_newstat[0]
            a_newstats.append(a_newstat)
            h_newstats.append(h_newstat)
        else:
            table = page.find_all('div', {'class' : 'content desktop'})
            num_cols = table[i * 2].find_all('thead')
            real_num = num_cols[0].find_all('th')
            del real_num[0]
            a_newstat = ['NaN' for _ in range(len(real_num))]
            h_newstat = ['NaN' for _ in range(len(real_num))]
            a_newstats.append(a_newstat)
            h_newstats.append(h_newstat)
    return a_newstats, h_newstats

def parse_teamstats(a_newstats, h_newstats):
    real_stats = []
    for team_newstats in (a_newstats, h_newstats):
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
    a_team, h_team = [dest + ' ' + name for (dest, name) in zip(names, cities)]
    return a_team, h_team

def get_page(link):
    req = r.get(link).text
    p = bs4.BeautifulSoup(req, 'html.parser')
    return p

def get_espn_boxstats(page):
    fields = ['gamepackage-passing', 'gamepackage-rushing', 'gamepackage-receiving',
                'gamepackage-interceptions', 'gamepackage-fumbles', 'gamepackage-interceptions',
                'gamepackage-fumbles', 'gamepackage-defensive', 'gamepackage-kickReturns',
                'gamepackage-puntReturns', 'gamepackage-kicking', 'gamepackage-punting']
    stats = [page.find_all('div', {'id' : id}) for id in fields]
    return stats

def get_boxscore(link='https://www.espn.com/nfl/boxscore?gameId=401127863'):
    page = get_page(link)
    stats = get_espn_boxstats(page)
    a_team, h_team = espn_box_teamnames(page)
    newstats = []
    for stat in stats:
        newstat = stat[0].find_all('tr', {'class' : 'highlight'})
        newstats.append(newstat)

    a_newstats, h_newstats = newstats_to_team_stats(newstats, page)
    real_stats = parse_teamstats(a_newstats, h_newstats)

    real_stats.extend([a_team, h_team])
    return real_stats
