"""
major todo is fixing the time.sleep() calls to using selenium waits
"""
import time
from sips.lines.bov.better import bet_utils as bu


class Better:
    """
    Better is a class for interacting with the utilities functions for interacting with
    bovada using python selenium.

    opt. args
    driver - selenium webdriver
    log_in - bool. determines whether the login() function is called on init
    verbose - bool. extra print statements
    """

    def __init__(self, driver=None, log_in=False, verbose=True):
        """
        the log_in is sick but the captcha sucks, we are literally just fuelling waymo
        """
        if driver:
            self.driver = driver
        else:
            self.driver = bu.get_driver()
        if log_in:
            bu.login(self.driver)

        self.verbose = verbose
        self.to_place = {
            0: ("Los Angeles Chargers", 0, 150),
            1: ("Jacksonville Jaguars", 1, 250),
            2: ("Tennessee Titans", 2, 350),
        }
        self.to_delete = [1, 2]
        self.bet_slip = []

        self.run()

    def run(self):
        """
        run() adds self.to_place into bovada betslip
        then the bets are deleted and removed from self.bet_slip
        """
        self.place()
        bu.delete_bets(self.driver, self.to_delete)
        to_delete = [self.to_place[x] for x in self.to_delete]
        self.bet_slip = [bet for bet in self.bet_slip if bet not in to_delete]

    def place(self):
        """
        place reads each bet in self.to_place and calls bet() for each
        """
        for key, value in enumerate(self.to_place.values()):
            bu.bet(value, key, self.driver, verbose=self.verbose)
            self.bet_slip.append(value)

    def info(self):
        """
        display the bet current bet slip
        """
        print(f"initial self.to_place: {self.to_place}")
        for i, queued_bet in enumerate(self.bet_slip):
            print(f"bet[{i}]: {queued_bet[0]}")


def main():
    """
    initializes an instance of Better, prints the bet sleep and returns
    """
    bot = Better()
    bot.info()
    time.sleep(5)
    return bot


if __name__ == "__main__":
    RET = main()
    time.sleep(60)
    RET.driver.close()
