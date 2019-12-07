import pandas as pd

from sips.h import grab
from sips.macros import sports_ref as sref


def trade_summary():
    df = grab.get_table(sref.NBA_URL + "/friv/injuries.fcgi", ["injuries"])
    return df


if __name__ == "__main__":

    df = trade_summary()
    print(df)
