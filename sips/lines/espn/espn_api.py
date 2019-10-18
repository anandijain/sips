import requests as r
import time
import json

from sips.h import macros as m
espn = 'http://site.api.espn.com/apis/site/v2/sports/'

def req_events(sport='nfl'):
    try:
        sport = m.league_to_sport_and_league[sport]
    except KeyError:
        print('forcing nfl')
        sport = m.league_to_sport_and_league['nfl']
    espn_json = r.get(espn + sport + '/scoreboard').json()
    events = espn_json['events']
    return events


def events(events=None, sport='nfl'):
    if not events:
        events = req_events()
    game_data = []
    for event in events:
        print(event['id'])
        game = parse_event(event)
        game_data.append(game)
    return game_data


def parse_event(event):
    d = desc(event)
    c = clock(event)
    s = state(event)
    w = weather(event)
    o = odds(event)
    t = tickets(event)

    return d + c + s + w + o + t


def desc(event):
    # date, id
    date = event['date']
    game_id = event['id']
    desc = event['name']
    short_name = event['shortName']
    return [date, game_id]


def clock(event):
    # clock, period
    status = event['status']
    clock = status['clock']
    period = status['period']
    return [clock, period]


def state(event):
    # completed, detail, state
    meta = event['status']['type']
    completed = meta['completed']
    detail = meta['detail']
    state = meta['state']
    return [completed, detail, state]


def weather(event):
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


def odds(event):
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


def tickets(event):
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


def competitors(event):
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


def teams(event):
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


if __name__ == '__main__':
    evs = events()
