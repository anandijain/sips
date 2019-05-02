def _eq(odd):
    # to find the adjusted odds multiplier 
    # returns float
    if odd == 0:
        return 0
    if odd >= 100:
        return odd/100.
    elif odd < 100:
        return abs(100/odd)


def eq_to_odd(equity):
    if equity > 1:
        odd = 100 * equity
        return odd
    elif equity <= 1:
        odd = -100/equity
        return odd
    elif equity == 0:
        return 0


def act(a):
    # simple function to easily change the action number into a string
    # returns string
    if a == 0:
        return 'BOUGHT AWAY'
    elif a == 1:
        return 'BOUGHT HOME'
    elif a == 2:
        return 'SKIP'
    else: 
        return 'action outside of defined actions'


def net(bet, bet2):
    # given a Bet pair (bet + hedge)
    # bet.amt * _eq(bet.a) should be equal to bet2.amt * _eq(bet2.h)
    bet_sum = bet.amt + bet2.amt
    if bet.team == 0:
        return bet.amt * _eq(bet.a_odds) - bet2.amt
    else:
        return bet.amt * _eq(bet.h_odds) - bet2.amt


def net_score(pick, pick2):
    if pick.team == 0:
        init_diff = pick.a_pts - pick.h_pts
        final_diff = pick2.a_pts - pick2.h_pts
    elif pick.team == 1:
        init_diff = pick.h_pts - pick.a_pts
        final_diff = pick2.h_pts - pick2.a_pts

        return final_diff - init_diff


def bet_amt(money):
    # return 0.05 * money + 100  # 100 is arbitrary
    return 100


def hedge_amt(bet, cur_odds):
    # takes in Bet 1 and calculates the 
    if bet.team == 0:
        return (bet.amt * (_eq(bet.a_odds) + 1))/ (_eq(cur_odds[1]) + 1)
    else:
        return (bet.amt * (_eq(bet.h_odds) + 1)) / (_eq(cur_odds[0]) + 1)


def net_given_odds(bet, cur_odds):
    bet2_amt = hedge_amt(bet, cur_odds)
    bet_sum = bet.amt + bet2_amt
    if bet.team == 0:
        return bet.amt * _eq(bet.a_odds) - bet2_amt
    else:
        return bet.amt * _eq(bet.h_odds) - bet2_amt
