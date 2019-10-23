import time
import bs4

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from sips.lines.bov import bet_utils as u



class Better:
    def __init__(self):
        self.driver = get_driver()


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

    button = u.get_bet_button(driver, team_name, amt, mkt_type)
    button.send_keys("\n")
    time.sleep(1)
    u.set_wager(driver, 0, amt)

    # if verbose:
    #     print(f'betting: {amt} to win {win_amt}')

    return win_amt


def locate_btn(game, team_name, mkt_type, verbose=False):
    '''
    given selenium game, team_name, and mkt_type
    returns find the specific bet button for the givens 

    game: selenium obj
    team_name: str
    mkt_type: 0 is point spread, 1 is moneyline, and 2 is over/under
    '''
    bet_buttons = get_bet_buttons(game)
    index = u.btn_index(game, mkt_type, team_name)
    to_click = bet_buttons[index]

    if verbose:
        print(f'btn_index: {index}')

    return to_click


def bet_on_team(driver, amt, mkt_type=2, team_name='Washington Redskins', verbose=False):
    '''
    mkt_type: 0 is point spread, 1 is moneyline, 2 is over under
    '''
    games = get_games(driver, verbose=verbose)
    game = u.game_from_team_name(games, team_name=team_name, verbose=verbose)

    to_click = locate_btn(game, amt=amt, mkt_type=mkt_type,
                           team_name=team_name, verbose=verbose)
    # to_click.click()
    to_click.send_keys("\n")
    time.sleep(1)

    u.set_wager(driver, 0, amt)
    return game


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


def get_games(driver=None, verbose=False):
    '''
    uses selenium to find the 'coupon-content'(s)
    each game has a tag_name of 'section'
    '''
    if not driver:
        driver = get_driver()
        
    games = driver.find_elements_by_class_name(u.GAME_CLASS)
    
    if verbose:
        print(f'len(games): {len(games)}')
        _ = [print(g.text) for g in games]

    return games


def team_names_from_games(games, zip_out=False, verbose=False):
    dogs = []
    favs = []
    for game in games:
        dog, fav = u.teams_from_game(game, verbose=verbose)
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

    game_mkts = game.find_elements_by_class_name(u.MARKETS_CLASS)

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


def get_bet_buttons(element):
    '''
    for a selenium object, find all of the classes w/ name 'bet-btn'
    '''
    bet_buttons = element.find_elements_by_class_name(u.BET_BUTTON_CLASS)
    return bet_buttons


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
