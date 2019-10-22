import time
import bs4

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

ROOT_URL = 'https://www.bovada.lv/sports/'
LINES = 'coupon-content'
MARKETS = 'markets-container'
BET_BUTTON = 'bet-btn'

IGNORE_FLAG_ID = 'custom-checkbox'
ALERT_CLASS_NAME = 'sp-overlay-alert__button'

RISK_ID = 'default-input--risk'

def bet(driver, button, amt):
    '''

    '''
    button.click()
    wager_box = driver.find_element_by_id(RISK_ID)
    wager_box.send_keys(amt)

    return 


def accept_review_step_skip(driver):
    '''

    '''
    labels = driver.find_elements_by_tag_name('label')
    label = labels[7]
    # not working
    # input_box = driver.find_element_by_id(IGNORE_FLAG_ID)
    # label = input_box.find_element_by_tag_name('label')
    label.click()
    
    button = driver.find_element_by_class_name(ALERT_CLASS_NAME)
    button.click()


def get_lines(driver=None, verbose=False):
    '''
    uses selenium to find the 'coupon-content'(s)
    '''
    if not driver:
        driver = get_driver()
        
    lines = driver.find_elements_by_class_name(LINES)
    
    if verbose:
        print(f'len(lines): {len(lines)}')
        _ = [print(l.text) for l in lines]

    return lines


def main():
    '''

    '''
    driver = get_driver()
    time.sleep(2.5)
    buttons = get_buttons(driver)
    trigger_review_slip_alert(driver, buttons)
    bet(driver, buttons[0], 250)
    return driver

def get_buttons(driver=None, verbose=True):
    '''
    
    '''
    if not driver:
        driver = get_driver()

    lines = get_lines(driver, verbose=verbose)
    mkts = mkts_from_lines(lines, verbose=verbose)
    buttons = buttons_from_mkts(mkts, verbose=verbose)
    return buttons

def trigger_review_slip_alert(driver, buttons=None):
    '''

    '''
    if not buttons:
        buttons = get_buttons(driver, True)
    button = buttons[0]    
    button.click()
    time.sleep(1)
    accept_review_step_skip(driver)


def mkts_from_lines(lines, verbose=False):
    '''
    returns MARKETS class name with selenium
    '''
    mkts = []

    for line in lines:
        line_mkts = line.find_elements_by_class_name(MARKETS)
        mkts += line_mkts
        
    if verbose:
        print(f'len(mkts): {len(mkts)}')
        _ = [print(mkt.text) for mkt in mkts]
    return mkts


def test_button_counts():
    driver = get_driver()
    buttons = get_buttons(driver)
    all_btns = all_bet_buttons(driver)
    print(f'new fxn: len(btns): {len(all_btns)}')
    print(f'old fxn: len(btns): {len(buttons)}')


def all_bet_buttons(driver):
    bet_buttons = driver.find_elements_by_class_name(BET_BUTTON)
    return bet_buttons


def buttons_from_mkts(mkts, verbose=False):
    '''
    returns list of buttons for each market provided
    '''
    buttons = []
    for mkt in mkts:
        mkt_buttons = mkt.find_elements_by_class_name(BET_BUTTON)
        buttons += mkt_buttons
    if verbose:
        print(f'len(buttons): {len(buttons)}')
        _ = [print(button.text) for button in buttons]
    return buttons


def get_driver(sport='football/nfl'):

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--incognito")

    driver = webdriver.Chrome(chrome_options=chrome_options)
    driver.get(ROOT_URL + sport)
    return driver


def to_soup(driver):
    '''
    driver is selenium webdriver
    '''
    p = bs4.BeautifulSoup(driver.page_source, 'html.parser')
    return p


if __name__ == '__main__':
    d = main()
    time.sleep(5)
    d.close()