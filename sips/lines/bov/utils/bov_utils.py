"""
utils functions for bov.py
"""

import requests as r

import sips.h.grab as g
import sips.h.parse as p
from sips.macros import bov as bm
from sips.macros import macros as m
from sips.lines import lines as ll
from sips.lines.bov.utils import scores


HEADERS = {"User-Agent": "Mozilla/5.0"}

MKT_TYPE = {
    "Point Spread": "ps",
    "Runline": "ps",
    "Puck Line": "ps",
    "Moneyline": "ml",
    "Total": "tot",
}

TO_GRAB = {
    "ps": ["american", "handicap"],
    "ml": ["american"],
    "tot": ["american", "handicap"],
    "competitors": ["home", "id", "name"],
}

PRICE_LABELS = {
    "ps": ["a_ps", "h_ps", "a_hcap", "h_hcap"],
    "ml": ["a_ml", "h_ml"],
    "tot": ["a_tot", "h_tot", "a_hcap_tot", "h_hcap_tot", "a_ou", "h_ou"],
}


def reduce_mkt_type(market_desc):
    try:
        reduced = MKT_TYPE[market_desc]
    except KeyError:
        return "ml"
    return reduced


def get_event():
    """
    used for quick debug purposes

    """
    req = g.req_json(bm.BOV_URL + "basketball/nba")
    event = req[0]["events"][0]
    return event


def parse_display_groups(event):
    """
    given an event, it will parse all of the displaygroups
    returns a dictionary with a key for every group and the value is the data of
    each market in that group
    """
    comps = competitors(event)

    groups = event["displayGroups"]
    full_set = {}
    for group in groups:
        desc = group.get("description")
        if not desc:
            continue
        # cleaned = clean_desc(desc)
        data_dict = parse_display_group(group, comps)
        full_set[desc] = data_dict

    # print(f'full_set: {full_set}')
    return full_set


def parse_display_group(display_group, competitors):
    """

    """
    group_markets = display_group.get("markets")
    data = parse_markets(group_markets, competitors=competitors)
    return data


def merge_lines_scores(lines, scores):
    """
    both type dict
    """
    ret = {}
    for k, v in lines.items():
        score_data = scores.get(k)
        if not score_data:
            score_data = [None for _ in range(5)]
        row = v[0] + score_data + v[1]
        ret[k] = row

    return ret


def get_links(sports, all_mkts=True):
    """

    """
    if all_mkts:
        links = [bm.BOV_URL + match_sport_str(s) for s in sports]
    else:
        links = filtered_links(sports)
    return links


def sports_to_jsons(sports, all_mkts=True):
    """

    """
    links = get_links(sports, all_mkts=all_mkts)
    jsons = g.reqs_json(links)
    return jsons


def sports_to_events(sports, all_mkts=False, verbose=False):
    """

    """
    jsons = sports_to_jsons(sports=sports, all_mkts=all_mkts)
    events = events_from_jsons(jsons)
    if verbose:
        print(f"events for sports: {sports}\n{events}")
    return events


def events_from_jsons(jsons):
    """
    jsons is a list of dictionaries for each sport 
    if rows, return parsed row data instead of list of events
    
    """
    events = [e for j in jsons for e in events_from_json(j)]
    return events


def rows_from_jsons(jsons):
    """
    jsons is a list of dictionaries for each sport 
    if rows, return parsed row data instead of list of events

    """
    events = [parse_event(e) for j in jsons for e in events_from_json(j)]
    return events


def dict_from_events(events, key="id", rows=True):
    """
    returns a dictionary of (key, event) or (key, list)

    key must be in the event json data
    rows: (bool)
        - if true, set key-vals to rows
    """
    event_dict = {e[key]: parse_event(e) if rows else e for e in events}
    return event_dict


def parse_event(event, output="list", verbose=False):
    """
    parses an event with three markets (spread, ml, total)
    returns list of data following the order in header()

    """
    game_id, sport, live, num_markets, last_mod = p.parse_json(
        event, ["id", "sport", "live", "numMarkets", "lastModified"], output="list"
    )
    ev_teams = teams(event)
    if not ev_teams:
        return
    a_team, h_team = ev_teams
    game_start_time = event.get("startTime")
    display_groups = event.get("displayGroups")
    markets = [dg.get("markets") for dg in display_groups]

    events_all_mkts = parse_display_groups(event)
    game_lines = events_all_mkts.get("Game Lines")

    if not game_lines:
        (
            a_ps,
            h_ps,
            a_hcap,
            h_hcap,
            ps_M_live,
            a_team,
            a_ml,
            h_team,
            h_ml,
            ml_M_live,
            a_tot,
            h_tot,
            a_hcap_tot,
            h_hcap_tot,
            a_ou,
            h_ou,
            tot_M_live,
        ) = [None for _ in range(17)]
    else:
        ml_M = game_lines.get("Moneyline_ml_M")
        ps_M = game_lines.get("Point Spread_ps_M")
        tot_M = game_lines.get("Total_tot_M")

        # to fix
        if ml_M:
            a_ml, h_ml, ml_M_live = ml_M
        else:
            a_ml, h_ml, ml_M_live = [None for _ in range(3)]
        if ps_M:
            a_ps, h_ps, a_hcap, h_hcap, ps_M_live = ps_M
        else:
            a_ps, h_ps, a_hcap, h_hcap, ps_M_live = [None for _ in range(5)]
        if tot_M:
            a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou, tot_M_live = tot_M
        else:
            a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou, tot_M_live = [
                None for _ in range(7)
            ]

    score_url = bm.BOV_SCORES_URL + game_id
    score_data = g.req_json(score_url)
    quarter, secs, a_pts, h_pts, status = scores.score(score_data)

    ret = [
        sport,
        game_id,
        a_team,
        h_team,
        last_mod,
        num_markets,
        live,
        quarter,
        secs,
        a_pts,
        h_pts,
        status,
        a_ps,
        h_ps,
        a_hcap,
        h_hcap,
        a_ml,
        h_ml,
        a_tot,
        h_tot,
        a_hcap_tot,
        h_hcap_tot,
        a_ou,
        h_ou,
        game_start_time,
    ]

    return ret


def grab_row_from_markets(markets):
    """
    to be deprecated, only grabs the filtered mkts
    parse main markets (match ps, ml, totals) json in bov event
    
    """
    (
        a_ps,
        h_ps,
        a_hcap,
        h_hcap,
        a_ml,
        h_ml,
        a_tot,
        h_tot,
        a_hcap_tot,
        h_hcap_tot,
        a_ou,
        h_ou,
    ) = ["NaN" for _ in range(12)]
    for market in markets:
        desc = market.get("description")
        period_desc, abbrv, live = mkt_period_info(market)
        if period_desc == "Match":
            outcomes = market["outcomes"]
            if desc == "Point Spread":
                a_ps, h_ps, a_hcap, h_hcap = spread(outcomes)
            elif desc == "Moneyline":
                a_ml, h_ml = moneyline(outcomes)
            elif desc == "Total":
                a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou = total(outcomes)

    data = [
        a_ps,
        h_ps,
        a_hcap,
        h_hcap,
        a_ml,
        h_ml,
        a_tot,
        h_tot,
        a_hcap_tot,
        h_hcap_tot,
        a_ou,
        h_ou,
    ]
    return data


def parse_markets(markets, competitors, output="dict"):
    """
    parse markets in bov event
    keys of all_markets are: mkt_desc + reduced mkt type + period abbrv

    key examples
    Moneyline -> moneyline_ml_M
    Point Spread -> point_spread_ps_M
    First Team to reach 20 points -> first_team_to_reach_20_points_ml_M
    
    """
    all_markets = {}

    for market in markets:
        market_desc = market.get("description")
        mkt_type = reduce_mkt_type(market_desc)
        if not mkt_type or market_desc == "Futures":
            continue

        period_desc, abbrv, live = mkt_period_info(market)

        mkt_key = market_desc + "_" + mkt_type + "_" + abbrv

        market_data = parse_market(market, competitors=competitors)
        all_markets[mkt_key] = market_data

    if output == "list":
        return list(all_markets.values())

    return all_markets


def parse_market(market, competitors):
    """
    given: market in bovada sport json
    returns: dictionary w (field , field_value)
    
    """
    period_desc, abbrv, live = mkt_period_info(market)
    outcomes = market.get("outcomes")

    market_desc = market.get("description")
    mkt_type = reduce_mkt_type(market_desc)

    if mkt_type == "ps":
        data = spread(outcomes)
    elif mkt_type == "tot":
        data = total(outcomes)
    else:
        data = ml_from_outcomes(outcomes, competitors)

    data.append(live)
    return data


def ml_from_outcomes(outcomes, competitors):
    """
    returns a_ml, h_ml
    
    """
    a_ml, h_ml = None, None
    if competitors:
        t1 = competitors[0]
        t2 = competitors[1]
        if t1["home"]:
            h_team = t1
            a_team = t2
        else:
            a_team = t1
            h_team = t2

        for oc in outcomes:
            competitor_id = oc.get("competitorId")
            desc = oc.get("description")
            price = oc.get("price")
            american = price["american"]

            if competitor_id == a_team["id"] or desc == a_team["name"]:
                a_ml = american
            elif competitor_id == h_team["id"] or desc == h_team["name"]:
                h_ml = american

    return [a_ml, h_ml]


def mkt_period_info(market):
    """
    returns the desc, abbrev, and live 
    
    """
    period = market.get("period")
    to_grab = ["description", "abbreviation", "live"]
    period_info = p.parse_json(period, to_grab, "list")
    return period_info


def clean_desc(desc):
    """

    """
    to_replace = [("-", ""), ("  ", " "), (" ", "_")]
    ret = desc.lower()
    for tup in to_replace:
        ret = ret.replace(tup[0], tup[1])
    return ret


def spread(outcomes):
    """
    gets both teams spread data
    
    """
    a_ps, a_hcap, h_ps, h_hcap = [None for _ in range(4)]
    for outcome in outcomes:
        price = outcome["price"]
        if outcome["type"] == "A":
            a_ps, a_hcap = p.parse_json(price, TO_GRAB["ps"], "list")
        else:
            h_ps, h_hcap = p.parse_json(price, TO_GRAB["ps"], "list")

    return [a_ps, h_ps, a_hcap, h_hcap]


def moneyline(outcomes):
    """
    gets both teams moneyline
    
    """
    a_ml = None
    h_ml = None
    for outcome in outcomes:
        price = outcome["price"]
        if outcome["type"] == "A":
            a_ml = price["american"]
        else:
            h_ml = price["american"]
    return [a_ml, h_ml]


def total(outcomes):
    """
    gets the over_under
    limited to two outcomes currently
    
    """
    null_ret = [None for _ in range(6)]
    if not outcomes:
        return null_ret
    try:
        a_outcome = outcomes[0]
    except IndexError:
        return null_ret
    try:
        h_outcome = outcomes[1]
    except IndexError:
        h_tot, h_hcap_tot, h_ou = [None for _ in range(3)]
        return null_ret

    a_outcome = outcomes[0]
    a_ou = a_outcome.get("type")
    a_price = a_outcome.get("price")
    a_tot, a_hcap_tot = p.parse_json(a_price, TO_GRAB["tot"], "list")

    h_outcome = outcomes[1]
    h_ou = h_outcome.get("type")
    h_price = h_outcome.get("price")
    h_tot, h_hcap_tot = p.parse_json(h_price, TO_GRAB["tot"], "list")

    return [a_tot, h_tot, a_hcap_tot, h_hcap_tot, a_ou, h_ou]


def competitors(event, verbose=False):
    """
    keys: 'home' (bool), 'id' (str), 'name' (str)
    
    """
    comps = event.get("competitors")
    if not comps:
        return
    data = [p.parse_json(t, TO_GRAB["competitors"]) for t in comps]
    if verbose:
        print(f"competitors: {data}")
    return data  # list of two dictionaries


def teams(event):
    """
    returns away, home team names (str)
    
    """
    teams = competitors(event)
    if not teams or len(teams) != 2:
        return
    t1 = teams[0]
    t2 = teams[1]
    if t1["home"]:
        h_team = t1["name"]
        a_team = t2["name"]
    else:
        a_team = t1["name"]
        h_team = t2["name"]

    return a_team, h_team


def bov_team_ids(event):
    """
    get competitor ids
    
    """
    teams = event.get("competitors")
    if not teams or len(teams) != 2:
        return

    t1 = teams[0]
    t2 = teams[1]
    if t1["home"]:
        h_id = t1["id"]
        a_id = t2["id"]
    else:
        a_id = t1["id"]
        h_id = t2["id"]
    return a_id, h_id


def filtered_links(sports, verbose=False):
    """
    append market filter to each url

    """
    sfx = "?marketFilterId=def&lang=en"
    links = [bm.BOV_URL + match_sport_str(s) + sfx for s in sports]
    if verbose:
        print(f"bov_links: {links}")
    return links


def get_ids(events):
    # returns the ids for all bov events given
    ids = [e.get("id") for e in events if e.get("id")]
    return ids


def events_from_json(json_dict):
    """
    simply accesses the events in a single json
    """
    if not json_dict:
        return []
    events = []
    for group in json_dict:
        group_events = group["events"]
        events += group_events
    return events


def match_sport_str(sport="baseball/mlb"):
    """
    maps string to the url suffix to bovada api
    give sport='all' to get a list of all  
    """
    try:
        sport = m.SPORT_TO_SUFFIX[sport]
    except KeyError:
        pass
    return sport


def bov_all_dict(verbose=False):
    """

    """
    all_dict = {}
    req = r.get(
        "https: // www.bovada.lv/services/sports /"
        "event/v2/events/A/description/basketball/nba"
    ).json()
    es = req[0].get("events")
    for event in es:
        desc = event.get("description")
        if not desc:
            continue
        event_dict = parse_display_groups(event)
        all_dict[desc] = event_dict

    if verbose:
        print(f"desc: {desc}")
        print(f"all_dict: {all_dict}")

    return all_dict


if __name__ == "__main__":

    ll.main()
