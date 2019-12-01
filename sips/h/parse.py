"""
 mainly bs4 utilities for grabbing data from sites:
"""
import bs4


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


def comments(page):
    # finds all of the bs4.Comments for a page
    comments = page.findAll(text=lambda text: isinstance(text, bs4.Comment))
    return comments


def get_table(page, table_id):
    # given bs4 page and table id, finds table using bs4. returns soup table
    table = page.find("table", {"id": table_id})
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
