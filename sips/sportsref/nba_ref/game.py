from sips.h import grab
from sips.h import parse
from sips.sportsref import utils


def get_games_tables(sport: str) -> dict:
    games_tables = {
        "/boxscores/": [
            "line_score",
            "four_factors",
            f"box-{sport}-game-basic",
            f"box-{sport}-q1-basic",
            f"box-{sport}-q2-basic",
            f"box-{sport}-h1-basic",
            f"box-{sport}-q3-basic",
            f"box-{sport}-q4-basic",
            f"box-{sport}-h2-basic",
        ],
        "/boxscores/pbp/": ["st_0", "st_1", "st_2", "st_3", "st_4", "st_5", "pbp"],
        "/boxscores/shot-chart/": [f"shooting-{sport}",],
    }
    return games_tables


bs = 'https://www.basketball-reference.com/boxscores/201910220LAC.html'
pbp = 'https://www.basketball-reference.com/boxscores/pbp/201910220LAC.html'
sc = 'https://www.basketball-reference.com/boxscores/shot-chart/201910220LAC.html'
root = 'https://www.basketball-reference.com/'


def main(game_id='201910220LAC.html'):
    sfxs = ['boxscores/', 'boxscores/pbp/', 'boxscores/shot-chart/']
    links = [root + sfx + game_id for sfx in sfxs]
    game_dict = {}
    for link in links:
        g_id = utils.url_to_id(link)
        p = grab.comments(link)
        ts = p.find_all('table')
        for t in ts:
            t_id = t.get('id')
            if t_id is None:
                continue
            df = parse.get_table(p, t_id, to_pd=True)
            game_dict[g_id + '_' + t_id] = df
    return game_dict

if __name__ == "__main__":

    gd = main()

    print(gd)
