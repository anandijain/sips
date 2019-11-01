'''

major todo is fixing the time.sleep() calls to using selenium waits

'''
import time

from sips.lines.bov.utils import bet as u


def login(driver=None, captcha_wait=60):
    '''
    goes to bovadas login and gets you to captcha, then waits for a time
    '''
    if not driver:
        driver = u.base_driver()

    driver.get(u.ROOT_URL + '?overlay=login')
    time.sleep(0.5)

    inputs = driver.find_elements_by_tag_name('input')

    email_field = inputs[0]
    email_field.send_keys('dahmeh.almos@gmail.com')

    password_field = inputs[1]
    password_field.send_keys('BeautifulBeachesinLagos5')

    login_field = inputs[2]
    login_field.send_keys('\n')

    print(f'waiting for {captcha_wait} seconds for user to fill out captcha')
    time.sleep(captcha_wait)
    # driver.close()


def get_driver(sport='football/nfl', sleep=1.5):
    '''
    opens chrome to nfl lines
    '''

    driver = u.base_driver()
    driver.get(u.ROOT_URL + 'sports/' + sport)

    if sleep:
        time.sleep(sleep)

    return driver


def bet(bet_tup, index, driver=None, verbose=False):
    '''
    bet_tup consists of team_name, mkt_type, amt

    ideally input args are (driver, game_id, amt)
    and returns the index of the top-line where it is currently located
    on the bet slip
    '''
    team_name, mkt_type, amt = bet_tup
    button = u.find_bet_button(team_name, mkt_type, driver, verbose)
    button.send_keys("\n")
    time.sleep(1)
    u.set_wager(driver, index, amt)
    time.sleep(1)


def delete_bets(driver, indices):
    '''
    deletes the bets from the betslip based on the top-line indices you specify
    '''
    if isinstance(indices, int):
        indices = [indices]

    top_lines = driver.find_elements_by_class_name(u.TOP_LINE_CLASS)

    for i in indices:
        top_line = top_lines[i]
        delete_bet_button = top_line.find_element_by_tag_name('button')
        delete_bet_button.send_keys('\n')
        time.sleep(1)
        print(f'bet[{i}] deleted')


class Better:
    '''
    Better is a class for interacting with the utilities functions for interacting with
    bovada using python selenium.

    opt. args
    driver - selenium webdriver
    log_in - bool. determines whether the login() function is called on init
    verbose - bool. extra print statements
    '''

    def __init__(self, driver=None, log_in=False, verbose=True):
        '''
        the log_in is sick but the captcha sucks, we are literally just fuelling waymo
        '''
        if driver:
            self.driver = driver
        else:
            self.driver = get_driver()
        if log_in:
            login(self.driver)

        self.verbose = verbose
        self.to_place = {
            0: ('Los Angeles Chargers', 0, 150),
            1: ('Jacksonville Jaguars', 1, 250),
            2: ('Tennessee Titans', 2, 350)
        }
        self.to_delete = [1, 2]
        self.bet_slip = []

        self.run()

    def run(self):
        '''
        run() adds self.to_place into bovada betslip
        then the bets are deleted and removed from self.bet_slip
        '''
        self.place()
        delete_bets(self.driver, self.to_delete)
        to_delete = [self.to_place[x] for x in self.to_delete]
        self.bet_slip = [bet for bet in self.bet_slip if bet not in to_delete]

    def place(self):
        '''
        place reads each bet in self.to_place and calls bet() for each
        '''
        for key, value in enumerate(self.to_place.values()):
            bet(value, key, self.driver, verbose=self.verbose)
            self.bet_slip.append(value)

    def info(self):
        '''
        display the bet current bet slip
        '''
        print(f'initial self.to_place: {self.to_place}')
        for i, queued_bet in enumerate(self.bet_slip):
            print(f'bet[{i}]: {queued_bet[0]}')


def main():
    '''
    initializes an instance of Better, prints the bet sleep and returns
    '''
    bot = Better()
    bot.info()
    time.sleep(5)
    return bot


if __name__ == '__main__':
    RET = main()
    time.sleep(60)
    RET.driver.close()
