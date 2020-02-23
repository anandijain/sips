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

DATA_DIR = PARENT_DIR + "data/"

NBA_DATA = DATA_DIR + "nba/"
NFL_DATA = DATA_DIR + "nfl/"
MLB_DATA = DATA_DIR + "mlb/"

NBA_GAME_DATA = NBA_DATA + "games/"
NBA_PLAYER_DATA = NBA_DATA + "players/"

NFL_GAME_DATA = NFL_DATA + "games/"
NFL_PLAYER_DATA = NFL_DATA + "players/"

MLB_GAME_DATA = MLB_DATA + "games/"
MLB_PLAYER_DATA = MLB_DATA + "players/"
