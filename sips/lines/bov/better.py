import time
import bs4

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from sips.lines.bov import bet_utils as u



class Better:
    def __init__(self, log_in=False, make_bet=True):
        self.driver = get_driver()

        if log_in:
            login()
            
        if make_bet:
            bet('Los Angeles Chargers', 1, self.driver)
            bet('Jacksonville Jaguars', 0, self.driver)

def login(driver=None, captcha_wait=60):
    if not driver:
        driver = base_driver()

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


def bet(team_name, mkt_type, amt, driver=None, verbose=False):
    '''
    ideally input args are (driver, game_id, amt)
    '''

    button = u.find_bet_button(team_name, mkt_type, driver)
    button.send_keys("\n")
    time.sleep(1)
    u.set_wager(driver, 0, amt)

    # if verbose:
    #     print(f'betting: {amt} to win {win_amt}')

    return 

def delete_bets(driver, indices):
    '''

    '''
    if isinstance(indices, int):
        indices = [indices]
    
    top_lines = driver.find_elements_by_class_name(u.TOP_LINE_CLASS)
    for i in indices:
        tl = top_lines[i]
        delete_bet_button = tl.find_element_by_tag_name('button')
        delete_bet_button.send_keys('\n')
        # delete_bet_button.click()
        print(f'bet[{i}] deleted')


def buttons_from_mkts(mkts, verbose=False):
    '''
    returns list of buttons for each market provided
    '''
    buttons = []
    for mkt in mkts:
        mkt_buttons = mkt.find_elements_by_class_name(u.BET_BUTTON_CLASS)
        buttons += mkt_buttons
    if verbose:
        print(f'len(buttons): {len(buttons)}')
        _ = [print(button.text) for button in buttons]
    return buttons


def get_driver(sport='football/nfl', sleep=None):
    '''
    opens chrome to nfl lines
    '''

    driver = u.base_driver()
    driver.get(u.ROOT_URL + 'sports/' + sport)

    if sleep:
        time.sleep(sleep)

    return driver


def main():
    '''

    '''
    # login(driver)
    driver = get_driver()
    games = get_games(driver)
    team_names_from_games(games, verbose=True)

    buttons = get_buttons(driver)
    u.trigger_review_slip_alert(driver, buttons)

    for bet_num, b in enumerate(buttons):
        bet(driver, b, bet_num * 20, bet_num, verbose=True)

        # make two bets
        if bet_num > 1:
            break

    time.sleep(1)
    delete_bets(driver, [0])

    return driver


if __name__ == '__main__':
    d = get_driver(sleep=2.5)
    # d = main()
    time.sleep(5)

    # d.close()
