from sips.lines.bov.better import bet_utils as bu


def test_button_counts():
    driver = bu.get_driver()
    all_btns = bu.get_bet_buttons(driver)
    print(f"new fxn: len(btns): {len(all_btns)}")
