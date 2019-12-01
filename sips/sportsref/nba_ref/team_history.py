import pandas as pd

from sips.macros.sports import nba
from sips.h import grab
from sips.sportsref.nba_ref import teams


def get_histories():
    team_pages = teams.get_teams()
    team_histories = []
    for page in team_pages:
        tables = page.find_all("table")
        print(f'len(tables): {len(tables)}')
        for t in tables:
            data = pd.read_html(t.prettify())
            team_histories.append(data)
    return team_histories


if __name__ == "__main__":

    ths = get_histories()
    print(ths)
