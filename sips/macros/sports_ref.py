# macros.py for highly usable data
mlb_url = "https://www.baseball-reference.com/"
nba_url = "https://www.basketball-reference.com/"
nfl_url = "https://www.pro-football-reference.com/"
nhl_url = "https://www.hockey-reference.com/"
fb_url = "https://fbref.com/"


nba_ns = "https://www.basketball-reference.com"
nfl_ns = "https://www.pro-football-reference.com"
fb_ns = "https://fbref.com"
nhl_ns = "https://www.hockey-reference.com"
mlb_ns = "https://www.baseball-reference.com"

urls = {"mlb": mlb_url, "nba": nba_url, "fb": fb_url, "nfl": nfl_url, "nhl": nhl_url}

urls_ns = {"mlb": mlb_ns, "nba": nba_ns, "fb": fb_ns, "nfl": nfl_ns, "nhl": nhl_ns}


letters = [
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
