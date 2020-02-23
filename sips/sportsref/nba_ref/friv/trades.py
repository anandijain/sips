import pandas as pd

from sips.h import grab
from sips.macros import sports_ref as sref


def trade_summary():
    trades_link = sref.NBA_URL + "/friv/trades.fcgi"
    print(trades_link)
    p = grab.page(trades_link)
    t = p.find("table", {"id": "summary_matrix"})
    df = pd.read_html(t.prettify())
    return df


if __name__ == "__main__":

    df = trade_summary()
    print(df)
