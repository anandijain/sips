import pandas as pd

import sips
from sips.macros import sports_ref as sref


def trade_summary():
    p = sips.page(sref.NBA_URL + "/friv/trades.fcgi")
    t = p.find("table", {"id": "summary_matrix"})
    df = pd.read_html(t.prettify())
    return df


if __name__ == "__main__":

    df = trade_summary()
    print(df)
