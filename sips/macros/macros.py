# Heavily referenced data


SPORT_TO_SUFFIX = {
    "nfl": "football/nfl",
    "nba": "basketball/nba",
    "nhl": "/hockey/nhl",
    "mlb": "baseball/mlb",
    "college-football": "football/college-football",
    "college-basketball": "basketball/college-basketball",
}
import sips

PROJ_DIR = sips.__path__[0] + "/"
PARENT_DIR = PROJ_DIR + "../"

LINES_DIR = PROJ_DIR + "../data/lines/lines/"
NBA_GAME_DATA = PARENT_DIR + "data/nba/games/"
