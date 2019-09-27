# grabs bov and espn
import requests as r
import time

from sips.lines.espn_box import get_boxscore

import json

bov ="https://www.bovada.lv/services/sports/event/v2/events/A/description/football?marketFilterId=def&eventsLimit=50&lang=en"
bov_scores_url = "https://services.bovada.lv/services/sports/results/api/v1/scores/"
espn = "http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

# convert bov metadata into keys
# team names to start
# there will never be a conflict of multiple live games with the same teams

class BovEvent:
    def __init__(self, event):
        # [sport, game_id, a_team, h_team, cur_time,  a_ps, h_ps, a_ml, h_ml, a_tot, h_tot, a_ou, h_ou]
        pass


def bov_team_to_id(team_name):
    pass

def get_bov_games(config_path):
    with open(config_path) as config:
        conf = json.load(config)
    ids = conf['games'][0]['game_ids']
    bov_events = get_bov_events()
    ret = []
    for game_id in ids:
        for event in bov_events:
            if int(event['id']) == game_id:
                ret.append(bov_line(event))
    return ret

def get_and_compare():
    bov, espn = get_events()
    rows = match_events(bov, espn)
    return rows

def get_bov_events():
    bov_json = r.get(bov).json()
    bov_events = bov_json[0]['events']
    return bov_events

def get_espn_events():
    espn_json = r.get(espn).json()
    events = espn_json['events']
    return events

def get_events():
    bov_events = get_bov_events()
    espn_events = get_espn_events()
    return bov_events, espn_events

def bov_lines(events=None):
    if not events:
        events = get_bov_events()
    lines = []
    for event in events:
        lines.append(bov_line(event))
    return lines


def bov_line(event):
    '''
    [sport, game_id, a_team, h_team, cur_time, last_mod, num_markets, live],
    [quarter, secs, a_pts, h_pts, status], [
    a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot,
    a_hcap_tot, h_hcap_tot, a_ou, h_ou, game_start_time]
    '''
    sport = event['sport']
    game_id = event['id']
    a_team, h_team = bov_teams(event)
    cur_time = time.time()
    last_mod = event['lastModified']
    num_markets = event['numMarkets']
    live = event['live']
    quarter, secs, a_pts, h_pts, status = bov_score(game_id)


    display_groups = event['displayGroups'][0]
    markets = display_groups['markets']
    a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou = bov_markets(markets)

    game_start_time = event['startTime']

    ret = [sport, game_id, a_team, h_team, cur_time, last_mod, num_markets, live, \
        quarter, secs, a_pts, h_pts, status, \
        a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot, \
        a_hcap_tot, h_hcap_tot, a_ou, h_ou, game_start_time]

    return ret

def bov_teams(event):
    # returns away, home
    team_one = event['competitors'][0]
    team_two = event['competitors'][1]
    if team_one['home']:
        h_team = team_one['name']
        a_team = team_two['name']
    else:
        a_team = team_one['name']
        h_team = team_two['name']
    return a_team, h_team


def bov_score(game_id):
    game_json = r.get(bov_scores_url + game_id).json()
    clock = game_json['clock']
    quarter = clock['periodNumber']
    secs = clock['relativeGameTimeInSecs']
    a_pts = game_json['latestScore']['visitor']
    h_pts = game_json['latestScore']['home']
    status = 0
    if game_json['gameStatus'] == "IN_PROGRESS":
        status = 1
    return (quarter, secs, a_pts, h_pts, status)


def bov_markets(markets):
    for market in markets:

        desc = market['description']
        outcomes = market['outcomes']

        if desc == 'Point Spread':
            a_ps, h_ps, a_hcap, h_hcap = parse_bov_spread(outcomes)
        elif desc == 'Moneyline':
            a_ml, h_ml = parse_bov_ml(outcomes)
        elif desc == 'Total':
            a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou = parse_bov_tot(outcomes)

    return (a_ps, h_ps, a_hcap, h_hcap, a_ml, h_ml, a_tot, h_tot,
            a_hcap_tot, h_hcap_tot, a_ou, h_ou)


def parse_bov_spread(outcomes):
    for outcome in outcomes:
        price = outcome['price']
        if outcome['type'] == 'A':
            a_ps = price['american']
            a_hcap = price['handicap']
        else:
            h_ps = price['american']
            h_hcap = price['handicap']
    return a_ps, h_ps, a_hcap, h_hcap

def parse_bov_ml(outcomes):
    for outcome in outcomes:
        price = outcome['price']
        if outcome['type'] == 'A':
            a_ml = price['american']
        else:
            h_ml = price['american']
    return a_ml, h_ml

def parse_bov_tot(outcomes):
    a_price = outcomes[0]['price']
    h_price = outcomes[1]['price']
    a_tot = a_price['american']
    h_tot = h_price['american']
    a_hcap_tot = a_price['handicap']
    h_hcap_tot = h_price['handicap']
    a_ou = outcomes[0]['type']
    h_ou = outcomes[1]['type']
    return a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou

def espn_events(events=None):
    if not events:
        events = get_espn_events()
    game_data = []
    for event in events:
        print(event['id'])
        game = parse_espn_event(event)
        game_data.append(game)
    return game_data

def parse_espn_event(event):
    # date, game_id = espn_event_desc(event)
    # clock, period = espn_event_clock(event)
    # completed, detail, state = espn_event_state(event)
    # attendance, date = espn_event_time_attendance(event)
    # display_weather, condition_id, temp, hi_temp = espn_event_weather(event)
    # details, over_under, provider, priority = espn_event_odds(event)
    # seats_available, summary = espn_event_tickets(event)

    date = espn_event_desc(event)
    clock = espn_event_clock(event)
    state = espn_event_state(event)
    weather = espn_event_weather(event)
    odds = espn_event_odds(event)
    tickets = espn_event_tickets(event)

    return date + clock + state + weather + odds + tickets

def espn_event_desc(event):
    # date, id
    date = event['date']
    game_id = event['id']
    desc = event['name']
    short_name = event['shortName']
    return [date, game_id]

def espn_event_clock(event):
    # clock, period
    status = event['status']
    clock = status['clock']
    period = status['period']
    return [clock, period]

def espn_event_state(event):
    # completed, detail, state
    meta = event['status']['type']
    completed = meta['completed']
    detail = meta['detail']
    state = meta['state']
    return [completed, detail, state]

def espn_event_weather(event):
    # display_weather, condition_id, temp
    weather = event.get('weather')
    if weather:
        display_weather = weather['displayValue']
        condition_id = weather['conditionId']
        temp = weather.get('temperature')
        hi_temp = weather.get('highTemperature')
        if not temp:
            temp = 'NaN'

        if not hi_temp:
            hi_temp = 'NaN'
    else:
        display_weather, condition_id, temp, hi_temp = ['NaN' for _ in range(4)]

    return [display_weather, condition_id, temp, hi_temp]

def espn_event_odds(event):
    comps = event['competitions'][0]
    odds = comps.get('odds')
    if odds:
        odds = odds[0]
        details = odds['details']
        over_under = odds['overUnder']
        provider = odds['provider']['name']
        priority = odds['provider']['priority']
    else:
        details, over_under, provider, priority = ['NaN' for _ in range(4)]

    return [details, over_under, provider, priority]

def espn_event_tickets(event):
    comps = event['competitions'][0]
    attendance = comps['attendance']
    tickets = comps.get('tickets')

    if tickets:
        tickets = tickets[0]
        seats_available = tickets['numberAvailable']
        summary = tickets['summary']
        lo_price = summary.split('$')[1]
    else:
        seats_available, lo_price = 'NaN', 'NaN'

    return [attendance, seats_available, lo_price]

def espn_event_competitors(event):
    competitors = comps['competitors']

    for competitor in competitors:
        records = competitor['records']
        if competitor['homeAway'] == 'home':
            h_record = records[0]['summary']
            h_score = records['score']
            h_team = competitor['team']['displayName']
        else:
            a_record = records[0]['summary']
            a_score = records['score']
            a_team = competitor['team']['displayName']

    return [a_record, h_record, a_score, h_score, a_team, h_team]

def espn_teams(event):
    # returns away, home
    team_one = event['competitions'][0]['competitors'][0]
    team_two = event['competitions'][0]['competitors'][1]
    if team_one['homeAway']:
        h_team = team_one['team']['displayName']
        a_team = team_two['team']['displayName']
    else:
        a_team = team_one['team']['displayName']
        h_team = team_two['team']['displayName']
    return a_team, h_team

def match_events(bov_events, espn_events):
    num_matched = 0
    rows = []
    for event in bov_events:
        bteams = bov_teams(event)
        for espn_event in espn_events:
            eteams = espn_teams(espn_event)

            if bteams == eteams:
                print(f'games matched: {bteams}')
                line = bov_line(event)
                espn_data = parse_espn_event(espn_event)
                row = line + espn_data
                rows.append(row)
                num_matched += 1
    print(f'len(bov_events): {len(bov_events)}\nlen(espn_events): {len(espn_events)}')
    print(f'num_matched: {num_matched}')
    return rows

def main():
    # b, e = get_events()
    # rows = match_events(b, e)
    stats = get_boxscore(link='https://www.espn.com/nfl/boxscore?gameId=401127863')
    return stats

if __name__ == "__main__":
    rows = main()
