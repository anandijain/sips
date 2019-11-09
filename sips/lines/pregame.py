import sips.lines.collate as col
# from nfl_ref import full_package as fp

if __name__ == '__main__':
    lines = col.get_and_compare(['football/nfl'])
    # df = fp.scraper_main()

    # print(f'df: {df}')
    # print(f'lines: {lines}')
