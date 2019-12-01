"""
 mainly bs4 utilities for grabbing data from sites:
"""
import pandas as pd
import bs4


def to_soup(html):
    """
    
    """
    soup = bs4.BeautifulSoup(html, "html.parser")
    return soup


def parse_json(json, keys, output="dict"):
    """
    input: dictionary and list of strings
    returns dict

    if the key does not exist in the json
    the key is still created with None as the value
    """
    data = {}
    json_keys = json.keys()
    for j_key in json_keys:
        if j_key in keys:
            d = json.get(j_key)
            data[j_key] = d

    if output == "list":
        return list(data.values())
    elif output == "dict":
        return data
    else:
        return None


def comments(page, verbose=False):
    # finds all of the bs4.Comments for a page
    comments = page.findAll(text=lambda text: isinstance(text, bs4.Comment))
    if verbose:
        for i, c in enumerate(comments):
            print(f'{i} : {c}')
    return comments


def get_table(page, table_id, to_pd=False):
    # given bs4 page and table id, finds table using bs4. returns soup table
    table = page.find("table", {"id": table_id})
    if not table:
        return
    if to_pd:
        table = pd.read_html(str(table))
    return table


def links(html, prefix=None):
    links = []
    a_tags = html.find_all("a")
    for a_tag in a_tags:
        link = a_tag['href']
        if prefix:
            link = prefix + link
        links.append(link)
    return links


if __name__ == "__main__":
    pass
