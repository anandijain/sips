from sips.sportsref import player


table_ids = [
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


if __name__ == "__main__":
    df = player.player_links('nhl', write=False)
    print(df)