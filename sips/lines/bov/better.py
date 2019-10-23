import time
import bs4

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from sips.lines.bov import bet_utils as u

'''

major todo is fixing the time.sleep() calls to using selenium waits

'''


def login(driver=None, captcha_wait=60):
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


def bet(team_name, mkt_type, amt, driver=None, verbose=False):
    '''
    ideally input args are (driver, game_id, amt)
    and returns the index of the top-line where it is currently located
    on the bet slip
    '''
    button = u.find_bet_button(team_name, mkt_type, driver, verbose)
    button.send_keys("\n")
    time.sleep(1)
    u.set_wager(driver, 0, amt)
    time.sleep(1)


def delete_bets(driver, indices):
    '''
    deletes the bets from the betslip based on the top-line indices you specify 
    '''
    if isinstance(indices, int):
        indices = [indices]
    
    top_lines = driver.find_elements_by_class_name(u.TOP_LINE_CLASS)

    for i in indices:
        tl = top_lines[i]
        delete_bet_button = tl.find_element_by_tag_name('button')
        delete_bet_button.send_keys('\n')
        time.sleep(1)
        print(f'bet[{i}] deleted')



class Better:
    def __init__(self, driver=None, test=True, verbose=True):
        '''
        betslip is a stack data structure
        '''
        if driver:
            self.driver = driver
        else:
            self.driver = get_driver()
            
        self.verbose = verbose
        self.to_place = {
            0: ('Los Angeles Chargers', 0, 150),
            1: ('Jacksonville Jaguars', 1, 250),
            2: ('Tennessee Titans', 2, 350)
        }
        self.to_delete = [1, 2]
        self.bet_slip = []

        if test:
            self.run()

    def run(self):
        self.place()
        delete_bets(self.driver, self.to_delete)
        self.bet_slip = [bet for bet in self.bet_slip if x not in self.to_delete]

    def place(self):
        for v in self.to_place.values():
            bet(v[0], v[1], v[2], self.driver, verbose=self.verbose)
            self.bet_slip.append(v)
    
    def info(self):
        print(f'self.to_place: {self.to_place}')
        for i, b in enumerate(self.bet_slip):
            print(f'bet[{i}]: {b[0]}')


def main():
    '''

    '''
    b = Better()
    b.info()
    time.sleep(5)
    return b

if __name__ == '__main__':
    b = main()
    time.sleep(5)
    b.driver.close()
