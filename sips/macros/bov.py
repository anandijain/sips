LINE_COLUMNS = [
    "sport",
    "game_id",
    "a_team",
    "h_team",
    "last_mod",
    "num_markets",
    "live",
    "quarter",
    "secs",
    "a_pts",
    "h_pts",
    "status",
    "a_ps",
    "h_ps",
    "a_hcap",
    "h_hcap",
    "a_ml",
    "h_ml",
    "a_tot",
    "h_tot",
    "a_hcap_tot",
    "h_hcap_tot",
    "a_ou",
    "h_ou",
    "game_start_time",
]

TO_SERIALIZE = [
    "sport",
    "a_team",
    "h_team",
    "last_mod",
    "num_markets",
    "live",
    "quarter",
    "secs",
    "a_pts",
    "h_pts",
    "status",
    "a_ml",
    "h_ml",
    "a_tot",
]


BOV_ROOT = "https://www.bovada.lv/"

# https://www.bovada.lv/services/sports/event/v2/events/A/description/
BOV_URL = BOV_ROOT + "services/sports/event/v2/events/A/description/"

# https://www.bovada.lv/services/sports/results/api/v1/scores/
BOV_SCORES_URL = BOV_ROOT + "services/sports/results/api/v1/scores/"

BOV_EVENT_SFX = "services/sports/event/coupon/events/A/description/"

SUFFIXES = [
    "?marketFilterId=def&liveOnly=true&lang=en",
    "?marketFilterId=def&preMatchOnly=true&eventsLimit=10&lang=en",
]


SPORTS = [
    "basketball",
    "baseball",
    "esports",
    "football",
    "tennis",
    "volleyball",
    "hockey",
    "badminton",
    "table-tennis",
]

TRANSITION_CLASS_STRINGS = [
    "a opens and h opens",
    "a opens and h goes up",
    "a opens and h goes down",
    "h opens and a goes up",
    "h opens and a goes down",
    "a closes and h closes",
    "a closes and h goes up",
    "a closes and h goes down",
    "h closes and a goes up",
    "h closes and a goes down",
    "stays same",
    "a goes up",
    "a goes down",
    "h goes up",
    "h goes down",
    "both a and h go up",
    "both a and h go down",
    "a goes up and h goes down",
    "a goes down and h goes up",
]


def build_urls(sports=SPORTS):
    """
    returns list of urls
    """
    urls = []
    for sport in sports:
        for suffix in SUFFIXES:
            urls.append(BOV_URL + sport + suffix)
    return urls


def build_url_dict(sports=SPORTS):
    """
    returns dictionary of {
        sport (str) : url (str)
    }
    """
    urls = {}
    for sport in sports:
        urls[sport] = [BOV_URL + sport + suffix for suffix in SUFFIXES]
    return urls
