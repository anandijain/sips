import bs4 

import pandas as pd
import requests as r

def main(years=(2000, 2019)):
    year_list = range(years[0], years[1] + 1)  # + 1 because range is not inclusve 
    dfs = []
    for year in year_list:
        dfs.append(get_df(year))
    print(f'Done: {len(dfs)} dataframes written')

def get_df(year, write=True):
    url = get_url(year=year)
    table = get_table(url)
    cols = get_headers(table.thead)
    player_ids = get_ids(table)
    raw_df = pd.read_html(table.prettify())[0]
    df = cat_ids(raw_df, player_ids)
    print(df)
    if write:
        fn = get_fn(year)
        df.to_csv(fn)
    return df 
        
def get_table(url):
    req = r.get(url)
    p = bs4.BeautifulSoup(req.text, "html.parser")
    table = p.find("table", {"id" : "combine"})
    return table

def get_headers(thead):
    header_cells = thead.find_all("th")
    cols = [header.text for header in header_cells]
    return cols

def get_url(year=2019):
    root = "https://www.pro-football-reference.com"
    url = "/draft/" + str(year) + "-combine.htm"
    return root + url

def get_ids(table):
    ids = []
    players = table.tbody.find_all('th', {'data-stat' : 'player'})
    for player in players:
        try:
            player_url = player.a['href']
            player_id = parse_id(player_url)
            ids.append(player_id)
        except TypeError:
            ids.append(player.text)
    return ids

def parse_id(player_url='/players/W/WoodZe00.htm'):
    ID = player_url.split("/")[3].split('.')[0]
    return ID

def cat_ids(raw_df, list_ids):
    id_col = pd.Series(list_ids, name="id")
    df = pd.concat([raw_df, id_col], axis=1)
    return df

def get_fn(year):
    return "./data/" + str(year) + "_nfl_combine.csv"
    

if __name__ == "__main__":
    main()


