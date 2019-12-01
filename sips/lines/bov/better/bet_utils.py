"""
utils for bovada selenium better.py program
"""
import time
import bs4

from selenium import webdriver

from sips.lines.bov import better

ROOT_URL = "https://www.bovada.lv/"
GAME_CLASS = "coupon-content"
MARKETS_CLASS = "markets-container"
BET_BUTTON_CLASS = "bet-btn"
SELECTED_BETS_CLASS = "bet-btn selected"

IGNORE_FLAG_ID = "custom-checkbox"
ALERT_CLASS_NAME = "sp-overlay-alert__button"

RISK_ID = "default-input--risk"
WIN_AMT_ID = "default-input--win"

RISK_AMT_DIV_NAME = "custom-field risk-field"
WIN_AMT_DIV_NAME = "custom-field win-field"

TOP_LINE_CLASS = "top-line"

MARKET_NAMES = {0: "point_spread", 1: "money_line", 2: "over_under"}


def get_games(driver, verbose=False):
    """
    uses selenium to find the 'coupon-content'(s)
    each game has a tag_name of 'section'
    """

    games = driver.find_elements_by_class_name(GAME_CLASS)

    if verbose:
        print(f"len(games): {len(games)}")
        _ = [print(g.text) for g in games]

    return games


def mkts_from_game(game, verbose=False):
    """
    gets tags of class name 'markets-container'
    """
    game_mkts = game.find_elements_by_class_name(MARKETS_CLASS)

    if verbose:
        _ = [print(mkt.text) for mkt in game_mkts]

    return game_mkts


def get_mkts(games, verbose=False):
    """
    returns MARKETS_CLASS class name with selenium
    """
    mkts = []

    for game in games:
        game_mkts = mkts_from_game(game, verbose=verbose)
        mkts += game_mkts

    if verbose:
        print(f"len(mkts): {len(mkts)}")
    return mkts


def buttons_from_mkts(mkts, verbose=False):
    """
    returns list of buttons for each market provided
    """
    buttons = []
    for mkt in mkts:
        mkt_buttons = mkt.find_elements_by_class_name(BET_BUTTON_CLASS)
        buttons += mkt_buttons
    if verbose:
        print(f"len(buttons): {len(buttons)}")
        _ = [print(button.text) for button in buttons]
    return buttons


def get_bet_buttons(element):
    """
    for a selenium object, find all of the classes w/ name 'bet-btn'
    """
    bet_buttons = element.find_elements_by_class_name(BET_BUTTON_CLASS)
    return bet_buttons


def bet_buttons_via_games(driver=None, verbose=False):
    """
    possibly equivalent to driver.find_elements_by_class_name('bet-btn')
    see get_bet_buttons in bet_utils.py
    """
    if not driver:
        driver = better.get_driver()

    games = get_games(driver, verbose=verbose)
    mkts = get_mkts(games, verbose=verbose)
    buttons = buttons_from_mkts(mkts, verbose=verbose)
    return buttons


def locate_btn(game, team_name, mkt_type, verbose=False):
    """
    given selenium game, team_name, and mkt_type
    returns find the specific bet button for the givens

    game: selenium obj
    team_name: str
    mkt_type: 0 is point spread, 1 is moneyline, and 2 is over/under
    """
    bet_buttons = get_bet_buttons(game)
    index = btn_index(game, mkt_type, team_name)
    to_click = bet_buttons[index]

    if verbose:
        print(f"btn_index: {index}")

    return to_click


def find_bet_button(team_name, mkt_type, driver, verbose=False):
    """
    finds the bet button related to the bet being made
    """
    games = get_games(driver)
    game = game_from_team_name(games, team_name, verbose=verbose)
    if not game:
        print(f"wtf")

    buttons = get_bet_buttons(game)
    index = btn_index(game, mkt_type, team_name)
    to_click = buttons[index]
    driver.execute_script("return arguments[0].scrollIntoView();", to_click)

    if verbose:
        print(f"game: {game}")
        print(f"len buttons: {len(buttons)}")
        for i, button in enumerate(buttons):
            print(f"button{[i]}: {button.text}")

    return to_click


def game_from_team_name(games, team_name, verbose=False):
    """
    rn just takes in 1 game => double header issue for mlb
    """
    for game in games:
        teams = teams_from_game(game)
        if team_name in teams:
            if verbose:
                print(f"found {team_name} game")
            return game
    if verbose:
        print(f"{team_name} game NOT found")
    return None


def team_names_from_games(games, zip_out=False, verbose=False):
    """
    given list of games (selenium), create a list of the team names
    """
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


def btn_index(game, mkt_type, team_name="Washington Redskins"):
    """
    since bovada has dogs above and favs below
    """
    teams = teams_from_game(game)

    if team_name == teams[0]:
        is_fav = False
    else:
        is_fav = True

    index = mkt_type * 2 + int(is_fav)
    return index


def teams_from_game(game, verbose=False):
    """
    game: selenium element
    dog, fav: both str
    """
    names = game.find_elements_by_tag_name("h4")
    dog = names[0].text
    fav = names[1].text

    if verbose:
        print(f"dog: {dog}")
        print(f"fav: {fav}\n")
    return dog, fav


def set_wager(driver, index, amt):
    """
    assumes that there is something in the betslip
    """

    wager_boxes = driver.find_elements_by_id(RISK_ID)

    try:
        wager_box = wager_boxes[index]
    except IndexError:
        accept_review_step_skip(driver)

        wager_boxes = driver.find_elements_by_id(RISK_ID)
        wager_box = wager_boxes[index]

    wager_box.send_keys(amt)


def trigger_review_slip_alert(driver, buttons=None):
    """
    program clicks a button to trigger the betslip alert,
    which it then accepts
    """

    if not buttons:
        buttons = bet_buttons_via_games(driver)
    print(f"buttons: {buttons}")
    button = buttons[0]
    button.send_keys("\n")
    # button.click()
    time.sleep(1)
    accept_review_step_skip(driver)


def accept_review_step_skip(driver):
    """
    accepts the review slip alert
    """
    labels = driver.find_elements_by_tag_name("label")
    label = labels[7]
    label.click()
    button = driver.find_element_by_class_name(ALERT_CLASS_NAME)
    button.send_keys("\n")
    time.sleep(1.5)


def get_team_names(driver):
    """
    gets the team names of all teams on current driver page
    """
    name_elements = driver.find_elements_by_class_name("name")
    team_names = [name.text for name in name_elements]
    return team_names


def login(driver=None, captcha_wait=60):
    """
    goes to bovadas login and gets you to captcha, then waits for a time
    """
    if not driver:
        driver = base_driver()

    driver.get(ROOT_URL + "?overlay=login")
    time.sleep(0.5)

    inputs = driver.find_elements_by_tag_name("input")

    email_field = inputs[0]
    email_field.send_keys("dahmeh.almos@gmail.com")

    password_field = inputs[1]
    password_field.send_keys("BeautifulBeachesinLagos5")

    login_field = inputs[2]
    login_field.send_keys("\n")

    print(f"waiting for {captcha_wait} seconds for user to fill out captcha")
    time.sleep(captcha_wait)
    # driver.close()


def get_driver(sport="football/nfl", sleep=1.5):
    """
    opens chrome to nfl lines
    """

    driver = base_driver()
    driver.get(ROOT_URL + "sports/" + sport)

    if sleep:
        time.sleep(sleep)

    return driver


def bet(bet_tup, index, driver=None, verbose=False):
    """
    bet_tup consists of team_name, mkt_type, amt

    ideally input args are (driver, game_id, amt)
    and returns the index of the top-line where it is currently located
    on the bet slip
    """
    team_name, mkt_type, amt = bet_tup
    button = find_bet_button(team_name, mkt_type, driver, verbose)
    button.send_keys("\n")
    time.sleep(1)
    set_wager(driver, index, amt)
    time.sleep(1)


def delete_bets(driver, indices):
    """
    deletes the bets from the betslip based on the top-line indices you specify
    """
    if isinstance(indices, int):
        indices = [indices]

    top_lines = driver.find_elements_by_class_name(TOP_LINE_CLASS)

    for i in indices:
        top_line = top_lines[i]
        delete_bet_button = top_line.find_element_by_tag_name("button")
        delete_bet_button.send_keys("\n")
        time.sleep(1)
        print(f"bet[{i}] deleted")


def base_driver(screen=None):
    """
    screen is either None, 'full', or 'min'
    """
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--incognito")

    driver = webdriver.Chrome(chrome_options=chrome_options)

    if not screen:
        pass
    elif screen == "full":
        driver.maximize_window()
    elif screen == "min":
        driver.minimize_window()

    return driver
