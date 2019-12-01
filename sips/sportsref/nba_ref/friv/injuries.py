import pandas as pd

import sips
from sips.macros import sports_ref as sref


def trade_summary():
    p = sips.get_page(sref.bk_url + "/friv/injuries.fcgi")
    t = p.find("table", {"id": "injuries"})
    df = pd.read_html(t.prettify())
    return df


if __name__ == "__main__":

    df = trade_summary()
    print(df)
