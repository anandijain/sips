"""
basic formulas and ops for data analysis

"""
from numbers import Number

from sips.h.bet_hedge import Bet, Hedge


def kelly(win_pct: float, eq_odd: float) -> float:
    return (win_pct * eq_odd - (1 - win_pct)) / eq_odd


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


def net(bet: Bet, bet2: Bet) -> Number:
    """
    bet_sum = bet.amt + bet2.amt
    
    """
    if bet.team == 0:
        return bet.amt * eq(bet.a_odds) - bet2.amt
    else:
        return bet.amt * eq(bet.h_odds) - bet2.amt


def profit(init_odds, cur_odds):
    """
    profit given initial and current odds

    """
    return eq(init_odds) - (eq(init_odds) + 1) / (eq(cur_odds) + 1)


def hedge_amt(bet, cur_odds):
    """
    amt to hedge for risk free profit

    """
    if bet.team == 0:
        return (bet.amt * (eq(bet.a_odds) + 1)) / (eq(cur_odds[1]) + 1)
    else:
        return (bet.amt * (eq(bet.h_odds) + 1)) / (eq(cur_odds[0]) + 1)


def net_given_odds(bet: Bet, cur_odds):
    """
    given a bet:Bet and the current odds, calculate net

    """
    bet2_amt = hedge_amt(bet, cur_odds)
    # bet_sum = bet.amt + bet2_amt
    if bet.team == 0:
        return bet.amt * eq(bet.a_odds) - bet2_amt
    else:
        return bet.amt * eq(bet.h_odds) - bet2_amt
