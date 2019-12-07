def get_games_tables(sport: str) -> dict:
    games_tables = {
        "/boxscores/": [
            "line_score",
            "four_factors",
            "box-NYK-game-basic" "box-NYK-q1-basic",
            "box-NYK-q2-basic",
            "box-NYK-h1-basic",
            "box-NYK-q3-basic",
            "box-NYK-q4-basic",
            "box-NYK-h2-basic",
        ],
        "/boxscores/pbp/": ["st_0", "st_1", "st_2", "st_3", "st_4", "st_5", "pbp"],
        "/boxscores/shot-chart/": ["shooting-PHO",],
    }
