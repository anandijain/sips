import time

import bs4
import requests as r
from requests_futures.sessions import FuturesSession
from concurrent.futures import ThreadPoolExecutor

HEADERS = {'User-Agent': 'Mozilla/5.0'}


def init_csv(fn, header, close=True):
    '''

    '''
    f = open(fn, 'a')
    write_list(f, header)
    if close:
        f.close()
    else:
        return f


def write_list(file, list):
    length = len(list)
    for i, elt in enumerate(list):
        file.write(str(elt))
        if i == length - 1:
            file.write('\n')
        else:
            file.write(',')


def get_page(link, verbose=False):
    '''

    '''
    if verbose:
        print(f'link: {link}')

    req = r.get(link, headers=HEADERS).text
    p = bs4.BeautifulSoup(req, 'html.parser')
    return p


def get_pages(links, output='list', verbose=False):
    '''
    list/dict comprehension on get_page()
    dict output uses links given as keys
    '''
    if output == 'list':
        pages = [get_page(l) for l in links]
    elif output == 'dict':
        pages = {l: get_page(l) for l in links}
    else:
        print(f'get_pages output type: {output} unsupported')
    return pages


def req(url):
    '''
    to be depr, same as req_json, but with a lot of try/excepts
    '''
    try:
        req = r.get(url, headers=HEADERS, timeout=10)
    except ConnectionResetError:
        print('connection reset error')
        time.sleep(2)
        return
    except r.exceptions.Timeout:
        print('requests.exceptions timeout error')
        time.sleep(2)
        return
    except r.exceptions.ConnectionError:
        print('connectionerror')
        time.sleep(2)
        return
    try:
        return req.json()
    except ValueError:
        time.sleep(2)
        return


def req_json(url, sleep=0.5, verbose=False):
    '''
    requests the link, returns json
    '''
    try:
        req = r.get(url, headers=HEADERS)
    except:
        return None

    time.sleep(sleep)

    try:
        json_data = req.json()
    except:
        print(f'{url} had no json')
        return None

    if verbose:
        print(f"req'd url: {url}")
    return json_data


def async_req(links, session=None, max_workers=10):
    '''
    asyncronous request of list of links
    '''
    if not session:
        session = FuturesSession(
            executor=ThreadPoolExecutor(max_workers=max_workers))

    jsons = [session.get(link).result().json() for link in links]
    return jsons


def async_req_dict(links, key, session=None, max_workers=10):
    '''
    the key 
    '''
    if not session:
        session = FuturesSession(
            executor=ThreadPoolExecutor(max_workers=max_workers))
    raw = [session.get(link).result().json() for link in links]
    jsons = {game.get(key): game for game in raw}
    return jsons
