import time
import bs4

from selenium import webdriver
from selenium.webdriver.common.keys import Keys


ROOT_URL = 'https://www.bovada.lv/'
GAME_CLASS = 'coupon-content'
MARKETS_CLASS = 'markets-container'
BET_BUTTON_CLASS = 'bet-btn'
SELECTED_BETS_CLASS = 'bet-btn selected'

IGNORE_FLAG_ID = 'custom-checkbox'
ALERT_CLASS_NAME = 'sp-overlay-alert__button'

RISK_ID = 'default-input--risk'
WIN_AMT_ID = 'default-input--win'

RISK_AMT_DIV_NAME = 'custom-field risk-field'
WIN_AMT_DIV_NAME = 'custom-field win-field'

TOP_LINE_CLASS = 'top-line'


MARKET_NAMES = {
    0: 'point_spread',
    1: 'money_line',
    2: 'over_under'
}


def get_bet_buttons(element):
    '''
    for a selenium object, find all of the classes w/ name 'bet-btn'
    '''
    bet_buttons = element.find_elements_by_class_name(u.BET_BUTTON_CLASS)
    return bet_buttons
    

def to_soup(driver):
    '''
    driver is selenium webdriver
    '''
    p = bs4.BeautifulSoup(driver.page_source, 'html.parser')
    return p

def get_bet_button(driver=None, team_name='Washington Redskins', amt=20, mkt_type=2, verbose=False):
    gs = get_games(driver)
    g = game_from_team_name(gs, team_name=team_name, verbose=True)
    buttons = get_bet_buttons(g)

    if verbose:
        print(f'game: {g}')
        print(f'len buttons: {len(buttons)}')
        for i, button in enumerate(buttons):
            print(f'button{[i]}: {button.text}')

    index = u.btn_index(g, mkt_type, team_name)
    to_click = buttons[index]
    driver.execute_script("return arguments[0].scrollIntoView();", to_click)
    return to_click


def game_from_team_name(games, team_name='Washington Redskins', verbose=False):
    '''
    rn just takes in 1 game => double header issue for mlb
    '''
    for game in games:
        teams = teams_from_game(game)
        if team_name in teams:
            if verbose:
                print(f'found {team_name} game')
            return game
    if verbose:
        print(f'{team_name} game NOT found')
    return None


def btn_index(game, mkt_type, team_name='Washington Redskins'):

    teams = teams_from_game(game)

    if team_name == teams[0]:
        is_fav = False
    else:
        is_fav = True

    index = mkt_type * 2 + int(is_fav)
    return index


def teams_from_game(game, verbose=False):
    names = game.find_elements_by_tag_name('h4')
    dog = names[0].text
    fav = names[1].text

    if verbose:
        print(f'dog: {dog}')
        print(f'fav: {fav}\n')
    return dog, fav


def set_wager(driver, index, amt):
    wager_boxes = driver.find_elements_by_id(RISK_ID)
    wager_box = wager_boxes[index]
    wager_box.send_keys(amt)


def trigger_review_slip_alert(driver, buttons=None):
    '''

    '''

    if not buttons:
        buttons = get_buttons(driver)
    print(f'buttons: {buttons}')
    button = buttons[0]
    button.send_keys('\n')
    # button.click()
    time.sleep(1)
    accept_review_step_skip(driver)


def accept_review_step_skip(driver):
    '''
    accepts the review slip alert
    '''
    labels = driver.find_elements_by_tag_name('label')
    label = labels[7]
    label.click()
    button = driver.find_element_by_class_name(u.ALERT_CLASS_NAME)
    button.send_keys('\n')


def get_team_names(driver):
    '''

    '''
    name_elements = driver.find_elements_by_class_name('name')
    team_names = [name.text for name in name_elements]
    return team_names


def base_driver(screen=None):
    '''
    screen is either None, 'full', or 'min'
    '''
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--incognito")

    driver = webdriver.Chrome(chrome_options=chrome_options)

    if not screen:
        pass
    elif screen == 'full':
        driver.maximize_window()
    elif screen == 'min':
        driver.minimize_window()

    return driver

