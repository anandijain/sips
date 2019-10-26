import os.path
import time

import bs4
import requests as r

HEADERS = {'User-Agent': 'Mozilla/5.0'}


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
    DELAY = 0.05
    if verbose:
        print(f'link: {link}')

    req = r.get(link, headers=HEADERS).text
    p = bs4.BeautifulSoup(req, 'html.parser')
    time.sleep(DELAY)
    return p


def req(url):
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
