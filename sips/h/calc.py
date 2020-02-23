"""
basic formulas and ops for data analysis

"""
from numbers import Number


def kelly(win_pct: float, eq_odd: float) -> float:
    return (win_pct * eq_odd - (1 - win_pct)) / eq_odd


def odd_to_win_pct(odd) -> float:

    if odd >= 100:
        return 100/(odd + 100)
    elif odd <= -100:
        return odd/(odd - 100)
    else:
        return


def pct_to_odd(pct: float) -> float:
    # float between 0 and 1
    if pct >= 0.5:
        return (pct/(1 - pct)) * -100
    elif pct < 0.5:
        return ((1 - pct) / pct) * 100
    else:
        return


def deltas(odds: list) -> list:
    """
    convolve over list returning difference at each point

    """
    delta_list = []
    prev = None
    for cur in odds:
        if not prev:
            prev = cur
            continue
        delta_list.append(cur - prev)
        prev = cur
    return delta_list


def eq(odd: Number) -> Number:
    """
    US odd -> decimal

    """
    if odd == 0:
        return 0
    elif odd >= 100:
        return odd / 100.0
    elif odd < 100:
        return abs(100 / odd)


def eq_to_odd(equity: float) -> Number:
    """
    Decimal -> US odd

    """
    if equity == 0:
        return 0.0
    elif equity > 1:
        odd = 100 * equity
        return odd
    elif equity <= 1:
        odd = -100 / equity
        return odd


def profit(init_odds, cur_odds):
    """
    profit given initial and current odds

    """
    return eq(init_odds) - (eq(init_odds) + 1) / (eq(cur_odds) + 1)


def test_odds_conversion(odds: list):
    for o in test_odds:
        e = eq(o)
        p = odd_to_win_pct(o)
        k = kelly(p + 0.05, e)
        print(f'odd: {o}, eq: {e}, pct:{p}, wager_pct: {k}')


if __name__ == "__main__":
    test_odds = [-1000, 500, 100, -100]
    test_odds_conversion(test_odds)