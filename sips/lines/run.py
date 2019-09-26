# grabs bov and espn
import requests as r

bov ="https://www.bovada.lv/services/sports/event/v2/events/A/description/football?marketFilterId=def&eventsLimit=50&lang=en"
espn = "http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

# convert bov metadata into keys
# team names to start
# there will never be a conflict of multiple live games with the same teams

def bov_team_to_id(team_name):
    pass

def fxn():
    bov, espn = get_events()
    compare(bov, espn)


def bov_events(bov_json):
    return events

def espn_events(espn_json):
    events = espn_json['events']
    return events

def get_events():
    bov_json = r.get(bov).json()
    espn_json = r.get(espn).json()
    bov_events = bov_json[0]['events']
    espn_events = espn_json['events']
    return bov_events, espn_events


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

def compare(bov_events, espn_events):
    num_matched = 0
    for event in bov_events:
        bteams = bov_teams(event)
        for espn_event in espn_events:
            eteams = espn_teams(espn_event)

            if bteams == eteams:
                print(f'games matched: {bteams}')
                num_matched += 1
    print(f'len(bov_events): {len(bov_events)}\nlen(espn_events): {len(espn_events)}')
    print(f'num_matched: {num_matched}')

def main():
    b, e = get_events()
    compare(b, e)

if __name__ == "__main__":
    main()
