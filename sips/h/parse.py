"""
 mainly bs4 utilities for grabbing data from sites
 
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


def comments(page, join=False, to_soup=False, verbose=False):
    # finds all of the bs4.Comments for a page
    comments = page.findAll(text=lambda text: isinstance(text, bs4.Comment))

    if join:
        comments = "".join(comments)
        if to_soup:
            comments = bs4.BeautifulSoup(comments, "html.parser")
    else:
        if to_soup:
            comments = [bs4.BeautifulSoup(c, "html.parser") for c in comments]

    if verbose:
        for i, c in enumerate(comments):
            print(f"$CS$: {i} : {c}")
    return comments


def get_table(page, table_id, find_all=False, to_pd=False):
    # given bs4 page and table id, finds table using bs4. returns soup table
    ret = []
    if find_all:
        table = page.find_all("table", {"id": table_id})
    else:
        table = page.find("table", {"id": table_id})

    if table is None:
        return

    if to_pd:
        if find_all:
            ret = [pd.read_html(t.prettify()) for t in table]
        else:
            ret = pd.read_html(table.prettify())[0]
    else:
        ret = table
    return ret


def links(html, prefix=None):
    links = []
    a_tags = html.find_all("a")
    for a_tag in a_tags:
        link = a_tag["href"]
        if prefix:
            link = prefix + link
        links.append(link)
    return links


if __name__ == "__main__":
    pass
