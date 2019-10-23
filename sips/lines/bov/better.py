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

class Better:
    def __init__(self):
        self.driver = get_driver()


def login(driver=None, captcha_wait=60):
    if not driver:
        driver = base_driver()

    driver.get(ROOT_URL + '?overlay=login')
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


def bet(driver, button, amt, index=0, verbose=False):
    '''
    ideally input args are (driver, game_id, amt)
    '''
    # button.click()
    button.send_keys("\n")
    time.sleep(1)

    set_wager(driver, index, amt)
    win_amt = driver.find_element_by_id(WIN_AMT_ID).text

    if verbose:
        print(f'betting: {amt} to win {win_amt}')

    return win_amt

def set_wager(driver, index, amt):
    wager_boxes = driver.find_elements_by_id(RISK_ID)
    wager_box = wager_boxes[index]
    wager_box.send_keys(amt)


def btn_index(game, mkt_type, team_name='Washington Redskins'):

    teams = teams_from_game(game)

    if team_name == teams[0]:
        is_fav = False
    else:
        is_fav = True

    index = mkt_type * 2 + int(is_fav)
    return index


def bet_on_game(game, amt, mkt_type=2, team_name='Washington Redskins', verbose=False):
    '''

    '''
    bet_buttons = get_bet_buttons(game)
    index = btn_index(game, mkt_type, team_name)
    to_click = bet_buttons[index]

    if verbose:
        print(f'btn_index: {index}')
        print(f'to_click.text: {to_click.text}')
        print(f'betting: {amt} on {team_name} at {time.asctime()}')

    return to_click


def bet_on_team(driver, amt, mkt_type=2, team_name='Washington Redskins', verbose=False):
    '''
    mkt_type: 0 is point spread, 1 is moneyline, 2 is over under
    '''
    games = get_games(driver, verbose=verbose)
    game = game_from_team_name(games, team_name=team_name, verbose=verbose)

    to_click = bet_on_game(game, amt=amt, mkt_type=mkt_type,
                           team_name=team_name, verbose=verbose)
    # to_click.click()
    to_click.send_keys("\n")
    time.sleep(1)

    set_wager(driver, 0, amt)
    return game


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


def delete_bets(driver, indices):
    '''

    '''
    if isinstance(indices, int):
        indices = [indices]
    
    top_lines = driver.find_elements_by_class_name(TOP_LINE_CLASS)
    for i in indices:
        tl = top_lines[i]
        delete_bet_button = tl.find_element_by_tag_name('button')
        delete_bet_button.send_keys('\n')
        # delete_bet_button.click()
        print(f'bet[{i}] deleted')


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
    button = driver.find_element_by_class_name(ALERT_CLASS_NAME)
    button.send_keys('\n')

def get_games(driver=None, verbose=False):
    '''
    uses selenium to find the 'coupon-content'(s)
    each game has a tag_name of 'section'
    '''
    if not driver:
        driver = get_driver()
        
    games = driver.find_elements_by_class_name(GAME_CLASS)
    
    if verbose:
        print(f'len(games): {len(games)}')
        _ = [print(g.text) for g in games]

    return games


def get_team_names(driver):
    name_elements = driver.find_elements_by_class_name('name')
    team_names = [name.text for name in name_elements]
    return team_names


def teams_from_game(game, verbose=False):
    names = game.find_elements_by_tag_name('h4')
    dog = names[0].text
    fav = names[1].text

    if verbose:
        print(f'dog: {dog}')
        print(f'fav: {fav}\n')
    return dog, fav


def team_names_from_games(games, zip_out=False, verbose=False):
    dogs = []
    favs = []
    for game in games:
        dog, fav = teams_from_game(game, verbose=verbose)
        dogs.append(dog)
        favs.append(fav)
    if zip_out:
        ret = list(zip(dogs, favs))
    else:
        ret = (dogs, favs)
    return ret


def get_buttons(driver=None, verbose=False):
    '''

    '''
    if not driver:
        driver = get_driver()

    games = get_games(driver, verbose=verbose)
    mkts = mkts_from_games(games, verbose=verbose)
    buttons = buttons_from_mkts(mkts, verbose=verbose)
    return buttons


def mkts_from_game(game, verbose=False):

    game_mkts = game.find_elements_by_class_name(MARKETS_CLASS)

    if verbose:
        _ = [print(mkt.text) for mkt in game_mkts]
    
    return game_mkts


def mkts_from_games(games, verbose=False):
    '''
    returns MARKETS_CLASS class name with selenium
    '''
    mkts = []

    for game in games:
        game_mkts = mkts_from_game(game, verbose=verbose)
        mkts += game_mkts
        
    if verbose:
        print(f'len(mkts): {len(mkts)}')
    return mkts


def test_button_counts():
    driver = get_driver()
    buttons = get_buttons(driver)
    all_btns = get_bet_buttons(driver)
    print(f'new fxn: len(btns): {len(all_btns)}')
    print(f'old fxn: len(btns): {len(buttons)}')


def get_bet_buttons(element):
    '''
    
    '''
    bet_buttons = element.find_elements_by_class_name(BET_BUTTON_CLASS)
    return bet_buttons


def buttons_from_mkts(mkts, verbose=False):
    '''
    returns list of buttons for each market provided
    '''
    buttons = []
    for mkt in mkts:
        mkt_buttons = mkt.find_elements_by_class_name(BET_BUTTON_CLASS)
        buttons += mkt_buttons
    if verbose:
        print(f'len(buttons): {len(buttons)}')
        _ = [print(button.text) for button in buttons]
    return buttons

def base_driver():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--incognito")

    driver = webdriver.Chrome(chrome_options=chrome_options)
    return driver


def get_driver(sport='football/nfl', minimize=False, sleep=None):

    driver = base_driver()
    driver.get(ROOT_URL + 'sports/' + sport)

    if minimize:
        driver.minimize_window()
    
    if sleep:
        time.sleep(sleep)

    return driver


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

    index = btn_index(g, mkt_type, team_name)
    to_click = buttons[index]
    driver.execute_script("return arguments[0].scrollIntoView();", to_click)
    return to_click

def quick_bet(team_name='Washington Redskins', amt=20, mkt_type=2, sport='football/nfl', driver=None, verbose=False):
    
    if not driver:
        driver = get_driver(sleep=2.5)
    # driver.fullscreen_window()
    # driver.minimize_window()

    # login(driver)
    driver.get(ROOT_URL + 'sports/' + sport)
    time.sleep(2.5)
    to_click = get_bet_button(driver, team_name, amt, mkt_type)
    to_click.send_keys('\n')
    try:
        set_wager(driver, 0, amt)    
    except:
        to_click.send_keys('\n')
        accept_review_step_skip(driver)
        time.sleep(0.5)
        set_wager(driver, 0, amt)    

    # bet_on_team(driver, 350)
    # return driver
    

def main():
    '''

    '''
    driver = get_driver()
    games = get_games(driver)
    team_names_from_games(games, verbose=True)

    buttons = get_buttons(driver)
    trigger_review_slip_alert(driver, buttons)

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
    d = quick_bet(driver=d)
    # d = main()
    time.sleep(5)

    # d.close()
