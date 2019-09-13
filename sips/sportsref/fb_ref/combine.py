import bs4 

import pandas as pd
import requests as r

def main(year=2019):
    url = get_url(year=year)
    table = get_table(url)
    cols = get_headers(table.thead)
    player_ids = get_ids(table)
    raw_df = pd.read_html(table.prettify())
    df = cat_ids(raw_df, player_ids)

    return df 

def get_table(url):
    req = r.get(url)
    p = bs4.BeautifulSoup(req.text, "html.parser")
    table = p.find("table", {"id" : "combine"})[0]
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

def table_to_df(table):
    df = pd.read_html(table.prettify())
    return df

def cat_ids(raw_df, list_ids):
    id_col = pd.Series(list_ids, name="id")
    df = pd.concat([raw_df, id_col], axis=1)
    return df



if __name__ == "__main__":
    main()


