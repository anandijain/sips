# macros.py for highly usable data
MLB_URL = "https://www.baseball-reference.com/"
NBA_URL = "https://www.basketball-reference.com/"
NFL_URL = "https://www.pro-football-reference.com/"
NHL_URL = "https://www.hockey-reference.com/"
FB_URL = "https://fbref.com/"


NBA_NS = "https://www.basketball-reference.com"
NFL_NS = "https://www.pro-football-reference.com"
FB_NS = "https://fbref.com"
NHL_NS = "https://www.hockey-reference.com"
MLB_NS = "https://www.baseball-reference.com"

URLS = {"mlb": MLB_URL, "nba": NBA_URL, "fb": FB_URL, "nfl": NFL_URL, "nhl": NHL_URL}

URLS_NS = {"mlb": MLB_NS, "nba": NBA_NS, "fb": FB_NS, "nfl": NFL_NS, "nhl": NHL_NS}


LETTERS = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "y",
    "z",
]  # no x


NFL_PLAYER_TABLE_IDS = [
    "stats",
    "rushing_and_receiving",
    "detailed_rushing_and_receiving",
    "returns",
    "defense",
    "scoring",
    "snap_counts",
    "all_pro",
    "fantasy",
    "combine",
]

NHL_PLAYER_TABLE_IDS = [
    "last5",
    "stats_basic_plus_nhl",
    "skaters_advanced",
    "stats_misc_plus_nhl",
    "stats_basic_plus_nhl_po",
    "stats_basic_minus_other",
    "sim_thru",
    "sim_career",
    "hat_tricks",
    "playoff_ot_goals",
]


NBA_PLAYER_TABLE_IDS = [
    "per_game",
    "totals",
    "per_minute",
    "per_poss",
    "advanced",
    "shooting",
    "pbp",
    "year-and-career-highs",
    "playoffs_per_game",
    "playoffs_totals",
    "playoffs_per_minute",
    "playoffs_per_poss",
    "playoffs_advanced",
    "playoffs_shooting",
    "playoffs_pbp",
    "year-and-career-highs-po",
    "all_star",
    "sim_thru",
    "all_college_stats",
    "sim_career",
    "all_salaries",
    "contracts_orl",
]


MLB_PLAYER_TABLE_IDS = [
    "batting_proj",
    "pitching_proj",
    "pitching_standard",
    "pitching_value",
    "pitching_postseason",
    "batting_standard",
    "batting_value",
    "batting_postseason",
    "standard_fielding",
    "appearances",
    "br-salaries",
]


def get_FB_PLAYER_TABLE_IDS(verbose=False) -> list:
    prefix = "stats_"
    sfxs = [
        "_ks_dom_lg",
        "_ks_dom_cup",
        "_ks_intl_cup",
        "_ks_expanded",
        "_ks_collapsed",
        "_dom_lg",
        "_dom_cup",
    ]
    categories = ["standard", "shooting", "passing", "playing_time", "misc"]
    table_ids = []
    for cat in categories:
        table_ids += [prefix + cat + sfx for sfx in sfxs]

    table_ids.append(prefix + "player_summary")

    if verbose:
        print(f"fb table ids: {table_ids}")

    return table_ids


FB_PLAYER_TABLE_IDS = get_FB_PLAYER_TABLE_IDS()

TABLE_IDS = {
    "mlb": MLB_PLAYER_TABLE_IDS,
    "nba": NBA_PLAYER_TABLE_IDS,
    "nfl": NFL_PLAYER_TABLE_IDS,
    "nhl": NHL_PLAYER_TABLE_IDS,
    "fb": FB_PLAYER_TABLE_IDS,
}
