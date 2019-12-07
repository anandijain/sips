"""
class for calculating reward for hedged bets

"""
from sips.h import calc


class Bet:
    """
    class storing bet info, will be stored in pair (hedged-bet)
    """

    def __init__(self, amt, action, odds, cur_state):
        """
        initialize bet

        """

        self.amt = amt
        self.team = action  # 0 for away, 1 for home
        self.a_odds = odds[0]
        self.h_odds = odds[1]
        self.cur_state = cur_state
        self.wait_amt = 0

    def reset_odds(self):
        # reset both odds
        self.a_odds = 0
        self.h_odds = 0

    def __repr__(self):
        # simple console log of a bet
        print(self.team)
        print("bet amt: " + str(self.amt) + " | team: " + str(self.team))
        print("a_odds: " + str(self.a_odds) + " | h_odds: " + str(self.h_odds))


class Hedge:
    """
    hedge takes in two bets of type Bet and returns the net reward

    """

    def __init__(self, bet, bet2):
        # input args is two Bets
        self.net = calc.net(bet, bet2)
        self.made_profit = self.net > 0
        self.bet = bet
        self.bet2 = bet2

    def __repr__(self):
        print("BET 1 of 2")
        self.bet.__repr__()
        print("BET 2 of 2")
        self.bet2.__repr__()
        print("hedged profit: " + str(self.net))
        print("steps waited: " + str(self.bet.wait_amt))
        if self.made_profit:
            print("$$$$$$$$$$$$$$$$$ made money")
        print("\n")
