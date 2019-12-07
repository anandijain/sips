"""
requesting and parsing data with requests, bs4, and pandas

"""
from concurrent.futures import ThreadPoolExecutor
import time

import pandas as pd
import bs4
import requests as r
from requests_futures.sessions import FuturesSession

from sips.h import parse

HEADERS = {"User-Agent": "Mozilla/5.0"}


def comments(link: str, verbose=False):
    """
    given a link, returns bs4 parsed html including commented sections

    """
    page_no_comments = page(link)
    page_comments = parse.comments(
        page_no_comments, join=True, to_soup=False, verbose=verbose
    )

    both = str(page_no_comments) + page_comments
    page_with_comments = bs4.BeautifulSoup(both, "html.parser")

    if verbose:
        print(f"page and comments: {page_with_comments}")

    return page_with_comments


def page(link: str):
    """
    request and bs4 parse html

    """
    req = req_text(link)
    soup = bs4.BeautifulSoup(req, "html.parser")
    return soup


def req_text(link: str) -> str:
    """
    request link and get text

    """
    req = r.get(link, headers=HEADERS).text
    return req


def pages(links, output="list", verbose=False):
    """
    Get a list of links using requests and bs4.

    Args:
        Required:
            links (list str): urls
        Optional:
            output ('list' | 'dict'): specify return dtype
            verbose (bool): print pages after retrieval 

    """
    if output == "list":
        pages = [page(l) for l in links]
    elif output == "dict":
        pages = {l: page(l) for l in links}
    else:
        pages = f"pages output type: {output} unsupported"

    if verbose:
        print(pages)

    return pages


def reqs_json(urls, sleep=0.5, verbose=False):
    """
    simple list concat on req_json

    """
    jsons = [req_json(url) for url in urls]
    return jsons


def req_json(url, sleep=0.5, verbose=False):
    """
    requests.get with some try excepts

    """

    try:
        req = r.get(url, headers=HEADERS, timeout=10)
    except:
        return None

    time.sleep(sleep)

    try:
        json_data = req.json()
    except:
        print(f"{url} had no json")
        return None

    if verbose:
        print(f"req'd url: {url}")
    return json_data


def async_req(links, output="list", session=None, max_workers=10, key=None):
    """
    asyncronous request of list of links

    Todo: depr

    """
    if not session:
        session = FuturesSession(executor=ThreadPoolExecutor(max_workers=max_workers))

    jsons = [session.get(link).result().json() for link in links]
    if output == "dict":
        if not key:
            print("no key provided, enumerating")
            jsons = {i: game for i, game in enumerate(jsons)}
        else:
            jsons = {game.get(key): game for game in jsons}

    return jsons


def get_table(link: str, table_ids: list, to_pd=True):
    """
    given a link, parses w/ bs4 and returns tables with table_id

    """

    tables = [
        pd.read_html(page(link).find("table", {"id": table_id}).prettify())
        if to_pd
        else page(link).find("table", {"id": table_id})
        for table_id in table_ids
    ]

    # handy
    if len(tables) == 1:
        tables = tables[0]

    return tables


def tables_from_links(links: str, table_ids: list, to_pd=True, flatten=False):
    """
    get tables from a list of links

    """

    all_tables = []
    for link in links:
        tables = [get_table(link, table_ids, to_pd=to_pd) for link in links]
        if flatten:
            all_tables += tables
        else:
            all_tables.append(tables)
    return tables


if __name__ == "__main__":
    pass
