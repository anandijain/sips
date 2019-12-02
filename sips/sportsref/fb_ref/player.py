from sips.sportsref import player


def t_ids(verbose=False) -> list:
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
        print(f'fb table ids: {table_ids}')

    return table_ids


if __name__ == "__main__":
    table_ids = t_ids()
    player.players('fb', table_ids)
