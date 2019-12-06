"""

"""
import sips.h.grab as g
from sips.lines import collate

espn = "http://site.api.espn.com/apis/site/v2/sports/"


def sport_to_api_url(sport="football/nfl"):
    url = espn + sport + "/scoreboard"
    return url


def req_events(sports=["football/nfl"]):
    urls = [sport_to_api_url(sport) for sport in sports]
    espn_jsons = g.reqs_json(urls)
    events = []
    for sport in espn_jsons:
        events += sport["events"]
    return events


def get_parsed_events(events=None, sports=["football/nfl"]):
    if not events:
        events = req_events(sports=sports)
    game_data = []
    for event in events:
        print(event["id"])
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
    teams = competitors(event)

    return d + c + s + w + o + t + teams


def desc(event):
    # date, id
    date = event["date"]
    game_id = event["id"]
    desc = event["name"]
    short_name = event["shortName"]
    return [date, game_id]


def clock(event):
    # clock, period
    status = event["status"]
    clock = status["clock"]
    period = status["period"]
    return [clock, period]


def state(event):
    # completed, detail, state
    meta = event["status"]["type"]
    completed = meta["completed"]
    detail = meta["detail"]
    state = meta["state"]
    return [completed, detail, state]


def weather(event):
    # display_weather, condition_id, temp
    weather = event.get("weather")
    if weather:
        display_weather = weather["displayValue"]
        condition_id = weather["conditionId"]
        temp = weather.get("temperature")
        hi_temp = weather.get("highTemperature")
        if not temp:
            temp = "NaN"

        if not hi_temp:
            hi_temp = "NaN"
    else:
        display_weather, condition_id, temp, hi_temp = ["NaN" for _ in range(4)]

    return [display_weather, condition_id, temp, hi_temp]


def odds(event):
    comps = event["competitions"][0]
    odds = comps.get("odds")
    if odds:
        odds = odds[0]
        details = odds["details"]
        over_under = odds["overUnder"]
        provider = odds["provider"]["name"]
        priority = odds["provider"]["priority"]
    else:
        details, over_under, provider, priority = ["NaN" for _ in range(4)]

    return [details, over_under, provider, priority]


def tickets(event):
    comps = event["competitions"][0]
    attendance = comps["attendance"]
    tickets = comps.get("tickets")

    if tickets:
        tickets = tickets[0]
        seats_available = tickets["numberAvailable"]
        summary = tickets["summary"]
        lo_price = summary.split("$")[1]
    else:
        seats_available, lo_price = "NaN", "NaN"

    return [attendance, seats_available, lo_price]


def competitors(event):
    competitions = event.get("competitions")
    if not competitions:
        print("couldnt find competitions")
        return

    competitors = competitions[0].get("competitors")

    for competitor in competitors:
        records = competitor["records"]
        all_splits = {"summary": None}
        for record in records:
            if record.get("name") == "All Splits":
                all_splits = record

        if competitor["homeAway"] == "home":
            h_record = all_splits["summary"]
            h_score = competitor["score"]
            h_team = competitor["team"]["displayName"]
        else:
            a_record = all_splits["summary"]
            a_score = competitor["score"]
            a_team = competitor["team"]["displayName"]

    return [a_record, h_record, a_score, h_score, a_team, h_team]


def teams(event):
    # returns away, home
    competitions = event.get("competitions")
    if not competitions:
        print("couldnt find competitions")
        return

    competitors = competitions[0].get("competitors")
    team_one = competitors[0]
    team_two = competitors[1]

    if team_one["homeAway"]:
        h_team = team_one["team"]["displayName"]
        a_team = team_two["team"]["displayName"]
    else:
        a_team = team_one["team"]["displayName"]
        h_team = team_two["team"]["displayName"]
    return a_team, h_team


if __name__ == "__main__":
    rows = collate.main()
