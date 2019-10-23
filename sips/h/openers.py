import os.path
import time

import bs4
import requests as r

headers = {'User-Agent': 'Mozilla/5.0'}

def write_list(file, list):
    length = len(list)
    for i, elt in enumerate(list):
        file.write(str(elt))
        if i == length - 1:
            file.write('\n')
        else:
            file.write(',')

def page(url):
    site = None
    i = 0
    while not site:
        try:
            site = r.get(url, headers=headers)
        except ConnectionError:
            i += 1
            time.sleep(2)
            if i == 5:
                return
    page_html = site.content
    page = bs4.BeautifulSoup(page_html, "html.parser")
    return page


def file(file_name):
    file = open(file_name, "w", encoding="utf-8")
    return file


def req(url):
    try:
        req = r.get(url, headers=headers, timeout=10)
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
