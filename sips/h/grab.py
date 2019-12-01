import time
import pandas as pd
import bs4
import requests as r
from requests_futures.sessions import FuturesSession
from concurrent.futures import ThreadPoolExecutor

HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_page(link):
    # request and bs4 parse html
    req = r.get(link, headers=HEADERS).text
    p = bs4.BeautifulSoup(req, "html.parser")
    return p


def get_pages(links, output="list", verbose=False):
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
        pages = [get_page(l) for l in links]
    elif output == "dict":
        pages = {l: get_page(l) for l in links}
    else:
        pages = f"get_pages output type: {output} unsupported"

    if verbose:
        print(pages)

    return pages


def reqs_json(urls, sleep=0.5, verbose=False):
    # simple list concat on req_json
    jsons = [req_json(url) for url in urls]
    return jsons


def req_json(url, sleep=0.5, verbose=False):
    # requests.get with some try excepts
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


def get_table(link, table_ids, to_pd=True):
    tables = [pd.read_html(get_page(link).find("table", {"id": table_id}).prettify()) if to_pd else get_page(
        link).find("table", {"id": table_id}) for table_id in table_ids]

    # handy
    if len(tables) == 1:
        tables = tables[0]
    return tables


def get_tables(links, table_ids, to_pd=True, flatten=False):
    """

    """
    all_tables = []
    for link in links:
        tables = [get_table(link, table_ids, to_pd=to_pd) for link in links]
        if flatten:
            all_tables += tables
        else:
            all_tables.append(tables)
    return tables