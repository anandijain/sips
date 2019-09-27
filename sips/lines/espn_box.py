import requests as r
import bs4

def espn_box_tds(tds):
    x = None
    prev = None
    data = []
    for td in tds:
        txt = td.text
        if txt == 'TEAM':
            x = 1
        if prev == 1:
            data.append(td.text)
        prev = x
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
    team_stats = espn_teamstats(page)
    real_stats = parse_teamstats(team_stats)
    real_stats.extend([a_team, h_team])
    return real_stats

if __name__ == "__main__":
    real_stats = get_boxscore()
    print(real_stats)
