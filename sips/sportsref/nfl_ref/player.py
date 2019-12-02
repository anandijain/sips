from sips.sportsref import player


table_ids = [
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


if __name__ == "__main__":
    # player.players("nfl", table_ids)
    df = player.player_links('nfl', write=False)
    print(df)
