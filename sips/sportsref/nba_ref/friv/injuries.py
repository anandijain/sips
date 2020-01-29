import pandas as pd

from sips.h import grab
from sips.macros import sports_ref as sref


def injuries():
    url = sref.NBA_URL + "/friv/injuries.fcgi"
    print(url)
    df = grab.get_table(url, ["injuries"])
    return df


if __name__ == "__main__":

    df = injuries()
    print(df)
