import time
import bs4

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

ROOT_URL = 'https://www.bovada.lv/sports/'
GAME_CLASS = 'coupon-content'
MARKETS_CLASS = 'markets-container'
BET_BUTTON_CLASS = 'bet-btn'

IGNORE_FLAG_ID = 'custom-checkbox'
ALERT_CLASS_NAME = 'sp-overlay-alert__button'

RISK_ID = 'default-input--risk'
WIN_AMT_ID = 'default-input--win'

RISK_AMT_DIV_NAME = 'custom-field risk-field'
WIN_AMT_DIV_NAME = 'custom-field win-field'

TOP_LINE_CLASS = 'top-line'

'''
rn games refered to by 


'''

def bet(driver, button, amt, index=0, verbose=False):
    '''
    ideally input args are (driver, game_id, amt)
    '''
    button.click()
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


def bet_on_team(driver, amt, mkt_type=2, team_name='Washington Redskins', verbose=False):
    '''
    mkt_type: 0 is point spread, 1 is moneyline, 2 is over under
    '''
    games = get_games(driver, verbose=verbose)
    game = game_from_team_name(games, team_name=team_name, verbose=verbose)
    teams = teams_from_game(game)
    if team_name == teams[0]:
        is_fav = False
    else:
        is_fav = True

    bet_buttons = get_bet_buttons(game)

    btn_index = mkt_type * 2 + int(is_fav)
    print(f'btn_index: {btn_index}')
    to_click = bet_buttons[btn_index]

    print(f'to_click.text: {to_click.text}')

    to_click.click()
    time.sleep(1)

    set_wager(driver, 0, amt)

    if verbose:
        print(f'betting: {amt} on {team_name} at {time.asctime()} in {teams} game')

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
        delete_bet_button.click()
        print(f'bet[{i}] deleted')


def trigger_review_slip_alert(driver, buttons=None):
    '''

    '''
    if not buttons:
        buttons = get_buttons(driver)

    button = buttons[0]
    button.click()
    time.sleep(1)
    accept_review_step_skip(driver)


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


def get_driver(sport='football/nfl', minimize=False, sleep=None):

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--incognito")

    driver = webdriver.Chrome(chrome_options=chrome_options)
    driver.get(ROOT_URL + sport)

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


def quick_bet(driver=None):
    
    if not driver:
        driver = get_driver(sleep=2.5)
    
    buttons = get_buttons(driver)
    trigger_review_slip_alert(driver, buttons)
    time.sleep(0.5)
    bet(driver, buttons[5], 350, index=0, verbose=False)
    # trigger_review_slip_alert(driver, buttons)

    time.sleep(0.5)
    g = bet_on_team(driver, 350)
    return g
    

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
    g = quick_bet(d)
    time.sleep(5)

    # d = main()
    # d.close()
