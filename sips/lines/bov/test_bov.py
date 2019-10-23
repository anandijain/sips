
def test_button_counts():
    driver = get_driver()
    buttons = get_buttons(driver)
    all_btns = get_bet_buttons(driver)
    print(f'new fxn: len(btns): {len(all_btns)}')
    print(f'old fxn: len(btns): {len(buttons)}')
