import requests as r
import time

import json

espn = "http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

def get_espn_events():
    espn_json = r.get(espn).json()
    events = espn_json['events']
    return events

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
